
import importlib
import os
import pickle
import warnings
from typing import Literal
from recbole_custom.data.dataloader import *
from recbole_custom.sampler import KGSampler, Sampler, RepeatableSampler
from recbole_custom.utils import ModelType, ensure_dir, get_local_time, set_color
from recbole_custom.utils.argument_list import dataset_arguments
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
from utils.utility.parser import parse_args
from recbole_custom.utils.mmhcl_utils import build_sim, build_knn_normalized_graph, get_u2u_mat
from recbole_custom.data.dataset.sequential_dataset import SequentialDataset

class MMHPDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.device = config['device']
        self.dataset_name = config['dataset']

        # MMHCL 参数
        self.top_k = config['top_k']
        self.image_feat_dim = config['image_feat_dim']
        self.text_feat_dim = config['text_feat_dim']

        # 容器
        self.image_features = None
        self.text_features = None
        self.i2i_mat = None
        self.u2u_mat = None

        # 1. 加载多模态特征 (MMHCL 逻辑)
        self._load_multimodal_features()

        # 2. 构建超图 (MMHCL 逻辑)
        self._build_graphs()

    def _load_multimodal_features(self):
        """
        加载 .npy 特征文件
        来源: MMHCL load_data.py
        """
        self.logger.info(set_color("Loading Multimodal Features...", 'pink'))

        # 路径构造
        data_root = self.config['data_path']
        img_file = os.path.join(data_root, self.dataset_name, 'image_feat.npy')
        txt_file = os.path.join(data_root, self.dataset_name, 'text_feat.npy')

        try:
            # 假设 npy 文件已经是按 RecBole 处理后的 token_id 排序好的
            # 或者是原始 ID 排序，且我们暂不处理复杂的 ID 映射 (简化 demo)
            raw_img = np.load(img_file)
            raw_txt = np.load(txt_file)

            # RecBole 的 Item ID 从 1 开始，0 是 padding
            # 所以我们需要在特征矩阵第 0 行拼一个全 0 向量
            padding_img = np.zeros((1, raw_img.shape[1]))
            padding_txt = np.zeros((1, raw_txt.shape[1]))

            # 拼接
            self.image_features = np.concatenate((padding_img, raw_img), axis=0)
            self.text_features = np.concatenate((padding_txt, raw_txt), axis=0)

            # 转 Tensor
            self.image_features = torch.FloatTensor(self.image_features).to(self.device)
            self.text_features = torch.FloatTensor(self.text_features).to(self.device)

            self.logger.info(f"Loaded Image Feat: {self.image_features.shape}")

        except FileNotFoundError:
            self.logger.warning("Multimodal files not found! Initializing random features.")
            # 随机初始化 (用于调试)
            if self.image_feat_dim is None:
                raise ValueError("image_feat_dim is None. Please set it in config.")

            self.image_features = torch.randn(
                self.item_num, self.image_feat_dim
            ).to(self.device)
            self.text_features = torch.randn(self.item_num, self.text_feat_dim).to(self.device)
            # 把 0 号 padding 置零
            self.image_features[0] = 0
            self.text_features[0] = 0

    def _build_graphs(self):
        """
        构建 i2i 和 u2u 图
        来源: MMHCL load_data.py & norm.py
        """
        self.logger.info(set_color("Building MMHP Graphs...", 'pink'))

        # --- 1. 构建 I2I 语义超图 ---
        # 对应 MMHCL: build_sim -> build_knn

        # 视觉 KNN
        sim_img = build_sim(self.image_features)
        # 屏蔽 padding
        sim_img[0, :] = 0;
        sim_img[:, 0] = 0
        graph_img = build_knn_normalized_graph(sim_img, self.top_k)

        # 文本 KNN
        sim_txt = build_sim(self.text_features)
        sim_txt[0, :] = 0;
        sim_txt[:, 0] = 0
        graph_txt = build_knn_normalized_graph(sim_txt, self.top_k)

        # 融合: 简单平均作为最终 i2i 邻接矩阵
        self.i2i_mat = (graph_img + graph_txt) / 2

        # --- 2. 构建 U2U 交互超图 ---
        # 对应 MMHCL: get_U2U_mat (A * A.T)
        self.u2u_mat = get_u2u_mat(self.inter_feat, self.user_num, self.item_num)

        # 转移至 GPU
        self.i2i_mat = self.i2i_mat.to(self.device)
        self.u2u_mat = self.u2u_mat.to(self.device)

        self.logger.info("Graphs built successfully.")
def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module("recbole_custom.data.dataset")
    if hasattr(dataset_module, config["model"] + "Dataset"):
        dataset_class = getattr(dataset_module, config["model"] + "Dataset")
    else:
        model_type = config["MODEL_TYPE"]
        type2class = {
            ModelType.GENERAL: "Dataset",
            ModelType.SEQUENTIAL: "SequentialDataset",
            ModelType.CONTEXT: "Dataset",
            ModelType.KNOWLEDGE: "KnowledgeBasedDataset",
            ModelType.TRADITIONAL: "Dataset",
            ModelType.DECISIONTREE: "Dataset",
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(
        config["checkpoint_dir"], f'{config["dataset"]}-{dataset_class.__name__}.pth'
    )
    file = config["dataset_save_path"] or default_file
    if os.path.exists(file):
        with open(file, "rb") as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ["seed", "repeatable"]:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color("Load filtered dataset from", "pink") + f": [{file}]")
            return dataset

    dataset = dataset_class(config)
    if config["save_dataset"]:
        dataset.save()
    return dataset

def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    ensure_dir(config["checkpoint_dir"])
    save_path = config["checkpoint_dir"]
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color("Saving split dataloaders into", "pink") + f": [{file_path}]")
    Serialization_dataloaders = []
    for dataloader in dataloaders:
        generator_state = dataloader.generator.get_state()
        dataloader.generator = None
        dataloader.sampler.generator = None
        Serialization_dataloaders += [(dataloader, generator_state)]

    with open(file_path, "wb") as f:
        pickle.dump(Serialization_dataloaders, f)


def load_split_dataloaders(config):
    """Load split dataloaders if saved dataloaders exist and
    their :attr:`config` of dataset are the same as current :attr:`config` of dataset.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        dataloaders (tuple of AbstractDataLoader or None): The split dataloaders.
    """

    default_file = os.path.join(
        config["checkpoint_dir"],
        f'{config["dataset"]}-for-{config["model"]}-dataloader.pth',
    )
    dataloaders_save_path = config["dataloaders_save_path"] or default_file
    if not os.path.exists(dataloaders_save_path):
        return None
    with open(dataloaders_save_path, "rb") as f:
        dataloaders = []
        for data_loader, generator_state in pickle.load(f):
            generator = torch.Generator()
            generator.set_state(generator_state)
            data_loader.generator = generator
            data_loader.sampler.generator = generator
            dataloaders.append(data_loader)

        train_data, valid_data, test_data = dataloaders
    for arg in dataset_arguments + ["seed", "repeatable", "eval_args"]:
        if config[arg] != train_data.config[arg]:
            return None
    train_data.update_config(config)
    valid_data.update_config(config)
    test_data.update_config(config)
    logger = getLogger()
    logger.info(
        set_color("Load split dataloaders from", "pink")
        + f": [{dataloaders_save_path}]"
    )
    return train_data, valid_data, test_data


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['[valid|test]_eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        model_type = config["MODEL_TYPE"]
        built_datasets = dataset.build()

        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )

        if model_type != ModelType.KNOWLEDGE:
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, shuffle=config["shuffle"]
            )
        else:
            kg_sampler = KGSampler(
                dataset,
                config["train_neg_sample_args"]["distribution"],
                config["train_neg_sample_args"]["alpha"],
            )
            train_data = get_dataloader(config, "train")(
                config, train_dataset, train_sampler, kg_sampler, shuffle=True
            )

        valid_data = get_dataloader(config, "valid")(
            config, valid_dataset, valid_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
        if config["save_dataloaders"]:
            save_split_dataloaders(
                config, dataloaders=(train_data, valid_data, test_data)
            )

    logger = getLogger()
    logger.info(
        set_color("[Training]: ", "pink")
        + set_color("train_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["train_batch_size"]}]', "yellow")
        + set_color(" train_neg_sample_args", "cyan")
        + ": "
        + set_color(f'[{config["train_neg_sample_args"]}]', "yellow")
    )
    logger.info(
        set_color("[Evaluation]: ", "pink")
        + set_color("eval_batch_size", "cyan")
        + " = "
        + set_color(f'[{config["eval_batch_size"]}]', "yellow")
        + set_color(" eval_args", "cyan")
        + ": "
        + set_color(f'[{config["eval_args"]}]', "yellow")
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.
    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        "MacridVAE": _get_AE_dataloader,
        "CDAE": _get_AE_dataloader,
        "ENMF": _get_AE_dataloader,
        "RaCT": _get_AE_dataloader,
        "RecVAE": _get_AE_dataloader,
    }

    if config["model"] in register_table:
        return register_table[config["model"]](config, phase)

    model_type = config["MODEL_TYPE"]
    if phase == "train":
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _get_AE_dataloader(config, phase: Literal["train", "valid", "test", "evaluation"]):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take 4 values: 'train', 'valid', 'test' or 'evaluation'.
            Notes: 'evaluation' has been deprecated, please use 'valid' or 'test' instead.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase not in ["train", "valid", "test", "evaluation"]:
        raise ValueError(
            "`phase` can only be 'train', 'valid', 'test' or 'evaluation'."
        )
    if phase == "evaluation":
        phase = "test"
        warnings.warn(
            "'evaluation' has been deprecated, please use 'valid' or 'test' instead.",
            DeprecationWarning,
        )

    if phase == "train":
        return UserDataLoader
    else:
        eval_mode = config["eval_args"]["mode"][phase]
        if eval_mode == "full":
            return FullSortEvalDataLoader
        else:
            return NegSampleEvalDataLoader


def _create_sampler(
    dataset,
    built_datasets,
    distribution: str,
    repeatable: bool,
    alpha: float = 1.0,
    base_sampler=None,
):
    phases = ["train", "valid", "test"]
    sampler = None
    if distribution != "none":
        if base_sampler is not None:
            base_sampler.set_distribution(distribution)
            return base_sampler
        if not repeatable:
            sampler = Sampler(
                phases,
                built_datasets,
                distribution,
                alpha,
            )
        else:
            sampler = RepeatableSampler(
                phases,
                dataset,
                distribution,
                alpha,
            )
    return sampler


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    train_neg_sample_args = config["train_neg_sample_args"]
    valid_neg_sample_args = config["valid_neg_sample_args"]
    test_neg_sample_args = config["test_neg_sample_args"]
    repeatable = config["repeatable"]
    base_sampler = _create_sampler(
        dataset,
        built_datasets,
        train_neg_sample_args["distribution"],
        repeatable,
        train_neg_sample_args["alpha"],
    )
    train_sampler = base_sampler.set_phase("train") if base_sampler else None

    valid_sampler = _create_sampler(
        dataset,
        built_datasets,
        valid_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    valid_sampler = valid_sampler.set_phase("valid") if valid_sampler else None

    test_sampler = _create_sampler(
        dataset,
        built_datasets,
        test_neg_sample_args["distribution"],
        repeatable,
        base_sampler=base_sampler,
    )
    test_sampler = test_sampler.set_phase("test") if test_sampler else None
    return train_sampler, valid_sampler, test_sampler


def get_bipartite_time_inter_mat(dataset, row='user', row_norm=True):
    from torch_geometric.utils import degree

    r"""Get the timebased bipartite interaction matrix of users and items.
    """
    if row == 'user':
        row_field, col_field = dataset.uid_field, dataset.iid_field
    else:
        row_field, col_field = dataset.iid_field, dataset.uid_field

    row = dataset.inter_feat[row_field]
    col = dataset.inter_feat[col_field]
    edge_index = torch.stack([row, col])

    if row_norm:
        deg = degree(edge_index[0], dataset.num(row_field))
        norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
        edge_weight = norm_deg[edge_index[0]]
    else:
        row_deg = degree(edge_index[0], dataset.num(row_field))
        col_deg = degree(edge_index[1], dataset.num(col_field))

        row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
        col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

        edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

    edge_time = dataset.inter_feat[dataset.time_field]

    return edge_index, edge_weight, edge_time
