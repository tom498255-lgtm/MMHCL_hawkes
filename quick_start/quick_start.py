import logging
from logging import getLogger
import sys
from recbole_custom.config import Config
from recbole_custom.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole_custom.data.transform import construct_transform
from recbole_custom.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

import logging
import sys
import os
from logging import getLogger

from recbole_custom.config import Config
from recbole_custom.utils import init_logger, init_seed, get_model, get_trainer
from recbole_custom.data import data_preparation, create_dataset

# ================= 新增：导入我们自定义的融合类 =================
from recbole_custom.model.sequential_recommender import MMHyperHawkes
from recbole_custom.data.utils import MMHPDataset
from recbole_custom.trainer.mmhp_trainer import MMHPTrainer


# ==============================================================

def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    r""" A fast running api, which includes the complete process of
    training and testing a model on a specified dataset
    """

    # 1. 配置初始化
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict
    )

    init_seed(config['seed'], config['reproducibility'])

    # 2. 日志初始化
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # 3. 数据集创建 (关键修改点)
    # ----------------------------------------------------------------
    logger.info("Loading Dataset...")
    if config['model'] == 'MMHyperHawkes':
        # 如果是 MMHP 模型，强制使用 MMHPDataset
        # 这个类包含了构建 i2i 和 u2u 超图的逻辑
        dataset = MMHPDataset(config)
    else:
        # 否则走 HyperHawkes 原有的通用逻辑
        dataset = create_dataset(config)
    # ----------------------------------------------------------------

    logger.info(dataset)

    # 4. 数据切分 (Train/Valid/Test)
    # RecBole 的 data_preparation 会自动处理 Dataset 对象
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # 5. 模型加载 (关键修改点)
    # ----------------------------------------------------------------
    logger.info("Loading Model...")
    init_seed(config['seed'] + config['local_rank'], config['reproducibility'])

    if config['model'] == 'MMHyperHawkes':
        # 强制实例化 MMHP 模型
        model = MMHyperHawkes(config, train_data.dataset).to(config['device'])
    else:
        # 否则走 HyperHawkes 原有的反射逻辑
        model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    # ----------------------------------------------------------------

    logger.info(model)

    # 6. 训练器加载 (关键修改点)
    # ----------------------------------------------------------------
    logger.info("Loading Trainer...")

    if config['model'] == 'MMHyperHawkes':
        # 强制使用包含 E-M 算法的 MMHPTrainer
        trainer = MMHPTrainer(config, model)
    else:
        # 否则走 HyperHawkes 原有的反射逻辑
        trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    # ----------------------------------------------------------------

    # 7. 模型训练
    logger.info("Start Training...")
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config['show_progress']
    )

    # 8. 模型评估
    logger.info("Start Testing...")
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config['show_progress']
    )

    # 记录结果
    logger.info(f"Best Valid Score: {best_valid_score}")
    logger.info(f"Test Result: {test_result}")

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }

def run_recboles(rank, *args):
    ip, port, world_size, nproc, offset = args[3:]
    args = args[:3]
    run_recbole(
        *args,
        config_dict={
            "local_rank": rank,
            "world_size": world_size,
            "ip": ip,
            "port": port,
            "nproc": nproc,
            "offset": offset,
        },
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    print(best_valid_score, best_valid_result, test_result)

    #tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered dataset, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - dataset (Dataset): The filtered dataset.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
