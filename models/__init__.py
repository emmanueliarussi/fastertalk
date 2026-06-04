from .stage1 import VQAutoEncoder
from .stage1_style import StyleVQAutoEncoder


def get_model(cfg):
    arch = getattr(cfg, "arch", "stage1")
    if arch == "stage1":
        return VQAutoEncoder(cfg)
    if arch == "stage1_style":
        return StyleVQAutoEncoder(cfg)
    raise ValueError(f"Unsupported arch '{arch}'.")
