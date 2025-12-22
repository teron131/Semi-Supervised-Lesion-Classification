from dataclasses import dataclass
from pathlib import Path
import random
import shutil

import albumentations as Augm
from albumentations.augmentations import ColorJitter, FancyPCA
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from .config import ensure_dirs, settings

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class DataTransforms:
    normalize: Augm.Compose
    geo: Augm.Compose
    color: Augm.Compose
    pca: Augm.Compose
    weak: Augm.Compose
    strong: Augm.Compose


def _read_rgb(image_filepath: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_filepath))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_filepath}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _has_images(directory: Path) -> bool:
    return any(directory.glob("*.jpg")) or any(directory.glob("*.jpeg")) or any(directory.glob("*.png"))


def _list_images(directory: Path) -> list[Path]:
    return sorted([path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])


def _label_from_path(image_filepath: str | Path) -> float:
    parent_dir = Path(image_filepath).parent.name
    return 1.0 if parent_dir == "malignant" else 0.0


def prepare_data(data_root: Path):
    """Split raw ISIC data into train, unlabeled, and val folders."""
    ensure_dirs()

    split_dirs = [
        settings.TRAIN_DIR / "benign",
        settings.TRAIN_DIR / "malignant",
        settings.UNLABELED_DIR,
        settings.VAL_DIR / "benign",
        settings.VAL_DIR / "malignant",
    ]
    if any(_has_images(directory) for directory in split_dirs):
        print("Split directories already contain images. Skipping data split.")
        return

    gt_path = data_root / "ISBI2016_ISIC_Part3_Training_GroundTruth.csv"
    if not gt_path.exists():
        print(f"Ground truth not found at {gt_path}. Skipping data split.")
        return

    df = pd.read_csv(gt_path, header=None, names=["image_id", "label"])
    img_lis = df["image_id"].tolist()
    lbl_lis = df["label"].tolist()

    N_total = len(img_lis)
    N_train = settings.TRAIN_SPLIT_SIZE
    N_unlabeled = settings.UNLABELED_SPLIT_SIZE
    if N_train + N_unlabeled > N_total:
        raise ValueError("Train + unlabeled split sizes exceed dataset size.")

    rng = np.random.default_rng(settings.SPLIT_SEED)
    shuffle_ix = rng.permutation(np.arange(N_total))
    ix_train = shuffle_ix[:N_train]
    ix_unlabeled = shuffle_ix[N_train : N_train + N_unlabeled]
    ix_val = shuffle_ix[N_train + N_unlabeled :]

    src_data_dir = data_root / "ISBI2016_ISIC_Part3_Training_Data"

    for i in ix_train:
        name, label = img_lis[i], lbl_lis[i]
        src_path = src_data_dir / f"{name}.jpg"
        if src_path.exists():
            shutil.copy2(src_path, settings.TRAIN_DIR / label / f"{name}.jpg")

    for i in ix_unlabeled:
        name = img_lis[i]
        src_path = src_data_dir / f"{name}.jpg"
        if src_path.exists():
            shutil.copy2(src_path, settings.UNLABELED_DIR / f"{name}.jpg")

    for i in ix_val:
        name, label = img_lis[i], lbl_lis[i]
        src_path = src_data_dir / f"{name}.jpg"
        if src_path.exists():
            shutil.copy2(src_path, settings.VAL_DIR / label / f"{name}.jpg")

    print(f"Data split completed: {len(ix_train)} train, {len(ix_unlabeled)} unlabeled, {len(ix_val)} val.")


def get_class_counts(train_dir: Path) -> tuple[int, int]:
    benign = len(_list_images(train_dir / "benign"))
    malignant = len(_list_images(train_dir / "malignant"))
    return benign, malignant


class AugmentedDataset(Dataset):
    """Dataset class that applies Albumentations transforms."""

    def __init__(self, images_filepaths: list[str | Path], transform: Augm.Compose | None = None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images_filepaths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        image_filepath = str(self.images_filepaths[idx])
        image = _read_rgb(image_filepath)

        # Label is 1.0 for malignant, 0.0 for benign based on folder name
        label = _label_from_path(image_filepath)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class UnlabeledDataset(Dataset):
    """Dataset class for unlabeled data with weak/strong views."""

    def __init__(self, root: str | Path, weak_transform: Augm.Compose, strong_transform: Augm.Compose):
        self.root = Path(root)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.samples = _list_images(self.root)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.samples[index]
        img = _read_rgb(path)
        weak = self.weak_transform(image=img)["image"]
        strong = self.strong_transform(image=img)["image"]
        return weak, strong

    def __len__(self) -> int:
        return len(self.samples)


def get_transforms() -> DataTransforms:
    """Define data transformations."""
    normalization = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.CenterCrop(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    geo_transform = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=1),
            Augm.RandomBrightnessContrast(p=0.1),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    col_transform = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            ColorJitter(),
            Augm.RandomBrightnessContrast(p=0.1),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    pca_transform = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            FancyPCA(),
            Augm.RandomBrightnessContrast(p=0.1),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    weak_transform = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=0.5),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    strong_transform = Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=0.5),
            Augm.OneOf(
                [
                    ColorJitter(),
                    FancyPCA(),
                    Augm.GaussianBlur(blur_limit=(3, 5)),
                ],
                p=0.8,
            ),
            Augm.CoarseDropout(p=0.5),
            Augm.RandomBrightnessContrast(p=0.2),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )

    return DataTransforms(
        normalize=normalization,
        geo=geo_transform,
        color=col_transform,
        pca=pca_transform,
        weak=weak_transform,
        strong=strong_transform,
    )


def _build_weighted_sampler(train_dataset: ConcatDataset, benign_count: int, malignant_count: int) -> WeightedRandomSampler:
    label_weights = []
    benign_weight = 1.0 / max(benign_count, 1)
    malignant_weight = 1.0 / max(malignant_count, 1)
    for dataset in train_dataset.datasets:
        if isinstance(dataset, AugmentedDataset):
            for filepath in dataset.images_filepaths:
                label_weights.append(malignant_weight if _label_from_path(filepath) == 1.0 else benign_weight)
    return WeightedRandomSampler(label_weights, num_samples=len(label_weights), replacement=True)


def get_dataloaders(batch_size: int = 32):
    """Prepare dataloaders for training, unlabeled, and validation sets."""
    transforms = get_transforms()

    # Paths
    benign_dir = settings.TRAIN_DIR / "benign"
    malignant_dir = settings.TRAIN_DIR / "malignant"

    # Get filepaths for upsampling minority class (malignant)
    benign_filepaths = _list_images(benign_dir)
    malignant_filepaths = _list_images(malignant_dir)

    # Upsampling (Malignant x2 as in notebook)
    aug_train_filepaths = [*benign_filepaths, *malignant_filepaths]
    if not settings.USE_WEIGHTED_SAMPLER:
        aug_train_filepaths = [*aug_train_filepaths, *malignant_filepaths]
    random.seed(settings.SPLIT_SEED)
    random.shuffle(aug_train_filepaths)

    # Create augmented datasets
    geo_ds = AugmentedDataset(aug_train_filepaths, transform=transforms.geo)
    col_ds = AugmentedDataset(aug_train_filepaths, transform=transforms.color)
    pca_ds = AugmentedDataset(aug_train_filepaths, transform=transforms.pca)
    aug_train_dataset = ConcatDataset([geo_ds, col_ds, pca_ds])

    standard_train_filepaths = [*benign_filepaths, *malignant_filepaths]
    standard_train_dataset = AugmentedDataset(standard_train_filepaths, transform=transforms.normalize)
    train_dataset = ConcatDataset([standard_train_dataset, aug_train_dataset])

    unlabeled_dataset = UnlabeledDataset(settings.UNLABELED_DIR, weak_transform=transforms.weak, strong_transform=transforms.strong)
    val_filepaths = _list_images(settings.VAL_DIR / "benign") + _list_images(settings.VAL_DIR / "malignant")
    val_dataset = AugmentedDataset(val_filepaths, transform=transforms.normalize)

    generator = torch.Generator().manual_seed(settings.SPLIT_SEED)
    sampler = None
    if settings.USE_WEIGHTED_SAMPLER:
        sampler = _build_weighted_sampler(train_dataset, len(benign_filepaths), len(malignant_filepaths))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.PIN_MEMORY,
        generator=generator,
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.PIN_MEMORY,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=settings.PIN_MEMORY,
        generator=generator,
    )

    return train_loader, unlabeled_loader, val_loader
