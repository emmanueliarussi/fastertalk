from .stage1 import VQAutoEncoder


def get_model(cfg):
    if cfg.arch != "stage1":
        raise ValueError(f"Unsupported arch '{cfg.arch}'. This minimal project supports only 'stage1'.")
    return VQAutoEncoder(cfg)
