import os
from typing import Callable

import numpy as np
import torch
from torchvision.models.resnet import ResNet18_Weights

from dataset import Classification, Dataset, GaussianQuantiles, Mnist, Moon, Random, cifar, imagenet
from torchays.cpa import Model, ProjectWrapper
from torchays.models import CIFARLinearNet, CIFARNet, LeNet, TestNetLinear, resnet

from .config import ANALYSIS, CIFAR10, COMMON, EXPERIMENT, GLOBAL, IMAGENET, MNIST, PATH, TESTNET, TOY, TRAIN, TYPE, configs
from .core import Analysis, Experiment, TrainHandler


def init_fun(seed: int = 0):
    def init():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    return init


def proj_net(
    new: Callable[[int, bool], Model],
    proj_dims: tuple | None = None,
    proj_values: torch.Tensor = None,
    handler: Callable[[str, torch.Tensor, str], None] = None,
):
    is_proj = (proj_dims and proj_values) is not None

    def wrapper(n_classes: int, training: bool = True):
        net = new(n_classes, training)
        if handler is not None and training:
            net.handler = handler
        if not is_proj or training:
            return net
        return ProjectWrapper(
            net,
            proj_dims=proj_dims,
            proj_values=proj_values,
        )

    return wrapper


def dataset():
    root: str = os.path.join(PATH.DIR, PATH.TAG) if PATH.TAG is not None and len(PATH.TAG) > 0 else PATH.DIR
    if GLOBAL.TYPE == TYPE.Moon:
        return Moon(root=root, n_samples=TOY.N_SAMPLES, noise=TOY.NOISE, random_state=COMMON.SEED, bias=TOY.BIAS)
    if GLOBAL.TYPE == TYPE.GaussianQuantiles:
        return GaussianQuantiles(root=root, n_samples=TOY.N_SAMPLES, n_classes=TOY.N_CLASSES, bias=TOY.BIAS, random_state=COMMON.SEED)
    if GLOBAL.TYPE == TYPE.Random:
        return Random(root=root, n_classes=TOY.N_CLASSES, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, random_state=COMMON.SEED, bias=TOY.BIAS)
    if GLOBAL.TYPE == TYPE.Classification:
        return Classification(root=root, n_samples=TOY.N_SAMPLES, in_features=TOY.IN_FEATURES, n_classes=TOY.N_CLASSES, bias=TOY.BIAS, random_state=COMMON.SEED)
    if GLOBAL.TYPE == TYPE.MNIST:
        return Mnist(root, MNIST.DOWNLOAD)
    if GLOBAL.TYPE == TYPE.CIFAR10:
        return cifar.Cifar10(root, CIFAR10.DOWNLOAD)
    if GLOBAL.TYPE == TYPE.ImageNet:
        return imagenet.ImageNet(root, IMAGENET.ROOT)
    if GLOBAL.TYPE == TYPE.TinyImageNet:
        return imagenet.TinyImageNet(root, IMAGENET.ROOT)
    return None


def net():

    def wrapper(n_classes: int, training: bool = True) -> Model:
        if GLOBAL.TYPE == TYPE.MNIST:
            # LeNet
            return LeNet()
        elif GLOBAL.TYPE == TYPE.CIFAR10:
            # CIFARNet
            if CIFAR10.LINEAR:
                return CIFARLinearNet(CIFAR10.NORM_LAYER)
            return CIFARNet(CIFAR10.NORM_LAYER)
        elif GLOBAL.TYPE == TYPE.ImageNet:
            weights = None
            if IMAGENET.PRETRAIN == True:
                weights = ResNet18_Weights.IMAGENET1K_V1
            return resnet.resnet18(
                weights=weights,
                num_classes=n_classes,
            )
        elif GLOBAL.TYPE == TYPE.TinyImageNet:
            return resnet.resnet18(
                num_classes=n_classes,
            )
        # Toy
        return TestNetLinear(
            in_features=TOY.IN_FEATURES,
            layers=TESTNET.N_LAYERS,
            n_classes=n_classes,
            norm_layer=TESTNET.NORM_LAYER,
        )

    return wrapper


def run(
    *,
    dataset: Dataset,
    net: Callable[[int, bool], Model],
    init_fun: Callable[[int], None] = init_fun,
    train_handler: TrainHandler = None,
):
    root_dir = os.path.join(dataset.path, GLOBAL.NAME)
    os.makedirs(root_dir, exist_ok=True)
    configs(root_dir)
    device = torch.device('cuda', COMMON.GPU_ID) if torch.cuda.is_available() else torch.device('cpu')
    exp = Experiment(
        root_dir=root_dir,
        net=net,
        dataset=dataset.make_dataset,
        init_fun=init_fun(COMMON.SEED),
        save_epoch=TRAIN.SAVE_EPOCH,
        device=device,
    )
    if TRAIN.TRAIN:
        exp.train(
            max_epoch=TRAIN.MAX_EPOCH,
            batch_size=TRAIN.BATCH_SIZE,
            lr=TRAIN.LR,
            train_handler=train_handler,
        )
    if EXPERIMENT.CPAS:
        exp.cpas(
            workers=EXPERIMENT.WORKERS,
            best_epoch=EXPERIMENT.WITH_BEST,
            bounds=EXPERIMENT.BOUND,
            depth=EXPERIMENT.DEPTH,
            is_draw=EXPERIMENT.WITH_DRAW,
            is_draw_3d=EXPERIMENT.WITH_DRAW_3D,
            is_draw_hpas=EXPERIMENT.WITH_DRAW_HPAS,
            is_statistic_hpas=EXPERIMENT.WITH_STATISTIC_HPAS,
        )
    if EXPERIMENT.POINT:
        exp.points(
            best_epoch=EXPERIMENT.WITH_BEST,
            bounds=EXPERIMENT.BOUND,
            depth=EXPERIMENT.DEPTH,
        )
    # run
    exp.run()

    # Analysis
    analysis = Analysis(
        root_dir=dataset.path,
        with_analysis=ANALYSIS.WITH_ANALYSIS,
        with_dataset=ANALYSIS.WITH_DATASET,
    )
    analysis()
