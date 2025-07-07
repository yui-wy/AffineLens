import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from numpy import sort
from PIL import Image
from torch.utils import data
from torchvision.datasets import imagenet
from torchvision.datasets.folder import IMG_EXTENSIONS, VisionDataset, default_loader, find_classes, has_file_allowed_extension, make_dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .dataset import Dataset


def _transform(img: Image.Image):
    trans = Compose(
        [
            Resize(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ],
    )
    return trans(img)


class _ImageNet(imagenet.ImageNet):
    def __init__(self, root, split="train", transform=None, target_transform=None, **kwargs):
        super().__init__(root, split, transform, target_transform, **kwargs)
        self.input_size = (3, 224, 224)


class ImageNet(Dataset):
    def __init__(self, root: str, data_root: str, split="train"):
        super().__init__("ImageNet", root)
        self.split = split
        self.data_root = data_root

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        try:
            imagenet = _ImageNet(root=self.data_root, split=self.split, transform=_transform)
            classes = len(imagenet.classes)
        except:
            imagenet, classes = None, 200
        return imagenet, classes

    def make_path(self):
        return os.path.join(self.root, self.type)


class _TinyImageNet(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.split = split
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        directory = os.path.expanduser(self.split_folder)
        classes, class_to_idx, class_to_label = self.find_classes(directory)
        idx_to_class = dict(zip(class_to_idx.values(), class_to_idx.keys()))

        self.samples = self.make_dataset(
            directory,
            class_to_idx=class_to_idx,
        )

        self.input_size = (3, 224, 224)
        self.loader = default_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.class_to_label = class_to_label

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def find_classes(self, directory: str) -> Tuple[List, Dict[str, int], Dict[str, str]]:
        class_to_label = self._find_class_to_label()
        find_classes = self._find_train_classes
        if self.split == "val":
            find_classes = self._find_val_classes
        classes, class_to_idx = find_classes(directory)
        return classes, class_to_idx, class_to_label

    def _find_train_classes(self, directory: str):
        return find_classes(directory)

    def _find_val_classes(self, directory: str):
        classes, class_to_idx = [], {}
        classes_set = set()
        annotations = self._read_val_annotations(directory)
        for annotation in annotations:
            classes_set.add(annotation[1])
        classes = sorted(list(classes_set))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _find_class_to_label(self) -> Dict[str, str]:
        set_nids = set()
        class_to_label: Dict[str, str] = {}
        wnids_file = os.path.join(self.root, "wnids.txt")
        words_file = os.path.join(self.root, "words.txt")
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                set_nids.add(entry.strip("\n"))
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] not in set_nids:
                    continue
                class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]
        return class_to_label

    def _read_val_annotations(self, directory: str) -> List[List[str]]:
        annotations = []
        with open(os.path.join(directory, "val_annotations.txt"), 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                annotations.append(words)
        return annotations

    def make_dataset(self, directory: Union[str, Path], class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        fcall = make_dataset
        if self.split == "val":
            fcall = self._make_val_dataset
        return fcall(directory, class_to_idx=class_to_idx, extensions=IMG_EXTENSIONS)

    def _make_val_dataset(
        self,
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: str | Tuple[str, ...] | None = None,
    ) -> List[Tuple[str, int]]:
        instances = []

        img_to_class = {}
        annotations = self._read_val_annotations(directory)
        for annotation in annotations:
            img_to_class[annotation[0]] = annotation[1]

        target_dir = os.path.join(directory, "images")
        if not os.path.isdir(target_dir):
            raise ValueError("'target_dir' is not a directory.")
        for root, _, files in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sort(files):
                path = os.path.join(root, fname)
                if not has_file_allowed_extension(path, extensions):
                    continue
                item = (path, class_to_idx[img_to_class.pop(fname)])
                instances.append(item)
        return instances

    def to_label(self, index: int):
        clz = self.idx_to_class.get(index)
        if clz is None:
            return None
        return self.class_to_label.get(clz)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class TinyImageNet(Dataset):
    def __init__(self, root: str, data_root: str, split="train"):
        super().__init__("TinyImageNet", root)
        self.split = split
        self.data_root = data_root

    def make_dataset(self) -> Tuple[data.Dataset, int]:
        try:
            imagenet = _TinyImageNet(root=self.data_root, split=self.split, transform=_transform)
            classes = len(imagenet.classes)
        except:
            imagenet, classes = None, 1000
        return imagenet, classes

    def make_path(self):
        return os.path.join(self.root, self.type)
