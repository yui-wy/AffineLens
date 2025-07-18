import json
import os
from typing import Any, Callable, Dict, List, Tuple

import torch

from torchays import nn
from torchays.models import resnet


class _config:
    @classmethod
    def __attr__(clz) -> Dict[str, Any]:
        attr: Dict[str, Any] = dict()
        for k, v in clz.__dict__.items():
            if k.startswith("_") or isinstance(v, torch.Tensor) or isinstance(v, Callable):
                continue
            if isinstance(v, type):
                v = v.__name__
            attr[k] = v
        return attr

    @classmethod
    def __set_attr__(clz, attr: Dict[str, Any]):
        for k, v in attr.items():
            setattr(clz, k, v)


__extend_config: List[_config] = list()


def extend(*config: _config):
    __extend_config.extend(config)


def __config() -> Tuple[_config]:
    configs = [ANALYSIS, CIFAR10, IMAGENET, COMMON, EXPERIMENT, GLOBAL, MNIST, PATH, TESTNET, TOY, TRAIN, TYPE]
    configs.extend(__extend_config)
    return configs


def check(file_dir: str):
    file = os.path.join(file_dir, "config.json")
    return os.path.exists(file)


def save(file_dir: str):
    attrs: Dict[str : Dict[str, Any]] = dict()
    for clz in __config():
        attrs[clz.__name__] = clz.__attr__()
    with open(os.path.join(file_dir, "config.json"), "w") as w:
        json.dump(attrs, w)


def load(file_dir: str):
    json_file = os.path.join(file_dir, "config.json")
    attrs: Dict[str : Dict[str, Any]] = dict()
    with open(json_file, "r") as r:
        attrs = json.load(r)
    for clz in __config():
        attr = attrs.get(clz.__name__, None)
        if attr is None:
            continue
        clz.__set_attr__(attr)


def print_cfg():
    print("-----------------------------------------------")
    print("Configuration:")
    print(f"Name: {GLOBAL.NAME}")
    print(f"Random Seed: {COMMON.SEED}")
    print(f"GPU_ID: {COMMON.GPU_ID}")
    print(f"Dataset: {GLOBAL.TYPE}")
    if GLOBAL.TYPE == TYPE.MNIST:
        print(f"Net: LeNet")
    elif GLOBAL.TYPE == TYPE.CIFAR10:
        print(f"Net: CIFARNet")
        print(f"|   Linear Model: {CIFAR10.LINEAR}")
        print(f"|   NormLayer: {CIFAR10.NORM_LAYER.__name__}")
    elif GLOBAL.TYPE == TYPE.ImageNet or GLOBAL.TYPE == TYPE.TinyImageNet:
        print(f"|   Data root: {IMAGENET.ROOT}")
        if IMAGENET.RESNET is None:
            print(print(f"Net: resnet18"))
        else:
            print(f"Net: {IMAGENET.RESNET.__name__}")
    else:
        print(f"|   n_samples: {TOY.N_SAMPLES}")
        print(f"|   n_class: {TOY.N_CLASSES}")
        print(f"|   in_feature: {TOY.IN_FEATURES}")
        if TESTNET.WITH_RESNET:
            print(f"Net: TestResNet")
            print(f"|   WITH_RESIDUAL: {TESTNET.WITH_RESIDUAL}")
        else:
            print(f"Net: TestNetLinear")
        print(f"|   Layers: {TESTNET.N_LAYERS}")
        print(f"|   NormLayer: {TESTNET.NORM_LAYER.__name__}")
    print(f"Action:")
    print(f"|   Train: {TRAIN.TRAIN}")
    print(f"|   CPAS: {EXPERIMENT.CPAS}")
    print(f"|       WORKS: {EXPERIMENT.WORKERS}")
    print(f"|       WITH_DRAW: {EXPERIMENT.WITH_DRAW}")
    print(f"|       WITH_DRAW_3D: {EXPERIMENT.WITH_DRAW_3D}")
    print(f"|       WITH_DRAW_HPAS: {EXPERIMENT.WITH_DRAW_HPAS}")
    print(f"|       WITH_STATISTIC_HPAS: {EXPERIMENT.WITH_STATISTIC_HPAS}")
    print(f"|       PROJ_DIM: {EXPERIMENT.PROJ_DIM}")
    print(f"|   Point: {EXPERIMENT.POINT}")
    print(f"|   Analysis: {ANALYSIS.WITH_ANALYSIS}")
    print(f"|   Draw dataset: {ANALYSIS.WITH_DATASET}")
    print("-----------------------------------------------")


def configs(dir: str):
    # Do not load configurations.
    # if check(dir):
    #     load(dir)
    save(dir)
    print_cfg()


class TYPE(_config):
    Moon = "Moon"
    GaussianQuantiles = "GaussianQuantiles"
    Random = "Random"
    Classification = "Classification"
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    ImageNet = "ImageNet"
    TinyImageNet = "TinyImageNet"


class COMMON(_config):
    GPU_ID: int = 7
    # random seed
    SEED: int = 5


class PATH(_config):
    TAG: str = "cpas-norm"
    DIR: str = os.path.join(os.path.abspath("./"), "cache")


class TOY(_config):
    # All toy datasets
    IN_FEATURES: int = 2
    N_SAMPLES: int = 100
    N_CLASSES: int = 2
    BIAS: float = 0
    # Moon
    NOISE: float = 0.2


class MNIST(_config):
    DOWNLOAD: bool = True


class CIFAR10(_config):
    DOWNLOAD: bool = True
    LINEAR: bool = False
    NORM_LAYER = nn.BatchNorm2d


class IMAGENET(_config):
    ROOT: str = "image-net"
    PRETRAIN: bool = False
    RESNET: Callable[..., resnet.ResNet] = None


class TESTNET(_config):
    N_LAYERS: List[int] = [64] * 3
    NORM_LAYER = nn.BatchNorm1d
    WITH_RESNET: bool = False
    WITH_RESIDUAL: bool = True


class GLOBAL(_config):
    NAME: str = "Linear-[64]x3-batch-no_l2"
    # ["Moon", "GaussianQuantiles", "Random", "Classification", "MNIST"]
    TYPE: str = "Random"


class TRAIN(_config):
    TRAIN: bool = True
    MAX_EPOCH: int = 10000
    SAVE_EPOCH: List[int] = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 500, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 20000, 30000]
    BATCH_SIZE: int = 64
    LR: float = 1e-3


class EXPERIMENT(_config):
    # cpas
    CPAS: bool = True
    # point
    POINT: bool = False
    # Project, "None"
    PROJ_DIM: Tuple[int] | None = None
    # The values of projection including "PROJ_DIM" which is useless.
    PROJ_VALUES: torch.Tensor = torch.zeros((1, 28, 28)) + 0.5
    # Bound for find cpa
    BOUND: Tuple[int] = (-1, 1)
    # The depth of the NN to draw
    DEPTH: int = -1
    # The number of the workers
    WORKERS: int = 1
    # With best epoch
    WITH_BEST: bool = False
    # Drawing
    # is drawing the region picture. Only for 2d input.
    WITH_DRAW: bool = True
    # is drawing the 3d region picture when "IS_DRAW" is True.
    WITH_DRAW_3D: bool = False
    # is handlering the hyperplanes arrangement.
    WITH_DRAW_HPAS: bool = False
    WITH_STATISTIC_HPAS: bool = False


class ANALYSIS(_config):
    # Analysis
    WITH_ANALYSIS: bool = False
    # draw the dataset distribution
    WITH_DATASET: bool = False
