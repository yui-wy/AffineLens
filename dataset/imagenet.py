import os
from typing import Tuple

from PIL import Image
from torch.utils import data
from torchvision.datasets import imagenet
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .dataset import Dataset


def _transform(img: Image.Image):
    trans = Compose(
        [
            Resize(224),
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ],
    )
    return trans(img)


class _ImageNet(imagenet.ImageNet):
    def __init__(self, root, split="train", transform=None, target_transform=None, **kwargs):
        super().__init__(root, split, transform, target_transform, **kwargs)
        self.input_size = (3, 224, 224)


class ImageNet(Dataset):
    def __init__(self, root: str, data_root: str, split="train"):
        super().__init__("imagenet", root)
        self.split = split
        self.data_root = data_root

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        try:
            imagenet = _ImageNet(root=self.data_root, split=self.split, transform=_transform)
            classes = len(imagenet.classes)
        except:
            imagenet, classes = None, 1000
        return imagenet, classes

    def make_path(self):
        return os.path.join(self.root, self.type)


class TinyImageNet(Dataset):
    def __init__(self, root: str, data_root: str, split="train"):
        super().__init__("imagenet", root)
        self.split = split
        self.data_root = data_root

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        try:
            imagenet = _ImageNet(root=self.data_root, split=self.split, transform=_transform)
            classes = len(imagenet.classes)
        except:
            imagenet, classes = None, 1000
        return imagenet, classes

    def make_path(self):
        return os.path.join(self.root, self.type)
