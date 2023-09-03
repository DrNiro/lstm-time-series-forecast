import argparse

from configurations.config import get_cfg_defaults


def cmd_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--symbol', type=str, default='GOOGL')
    parser.add_argument('--window_size', type=int, default=17)
    parser.add_argument('--updates', type=bool, default=False)

    return parser.parse_args()


def get_config():
    cfg = get_cfg_defaults()

    args = cmd_arguments()

    merge_args = []
    for arg in vars(args):
        merge_args.append(f'data.{arg}')
        merge_args.append(getattr(args, arg))

    cfg.merge_from_list(merge_args)
    cfg.freeze()

    return cfg

