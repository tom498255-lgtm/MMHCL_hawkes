import argparse
import os
import time
from recbole_custom.quick_start import run_recbole

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 默认模型修改为 MMHP
    parser.add_argument('--model', '-m', type=str, default='MMHyperHawkes', help='name of models')
    # 默认数据集修改为 tiktok (MMHCL 常用数据集)
    parser.add_argument('--dataset', '-d', type=str, default='tiktok', help='name of dataset')
    # 允许传入自定义配置文件路径
    parser.add_argument('--config_files', type=str, default='recbole_custom/config/mmhp.yaml', help='config files')
    parser.add_argument('--show-progress', '-sp', type=int, default=0)
    parser.add_argument('--seed', '-s', type=int, default=2023)

    args, _ = parser.parse_known_args()

    config_file_list = [f"./configs/general_full.yaml"]

    config_file_list.append(f"./configs/dataset/{args.dataset}.yaml")
    model_config_path = f"./configs/model/{args.model}/{args.dataset}.yaml"
    if os.path.exists(model_config_path):
        config_file_list.append(model_config_path)

    print(config_file_list)

    if args.seed is None or args.seed == 0:
        args.seed = int(time.time() // 1000)

    run_recbole(model=args.model, dataset=args.dataset,
                config_file_list=config_file_list,
                config_dict=vars(args))
    print(args.dataset)
    print(args)