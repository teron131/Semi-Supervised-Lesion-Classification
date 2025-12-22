import os
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


def _read_rgb(image_filepath: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_filepath))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_filepath}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def prepare_data(data_root: Path):
    """Split raw ISIC data into train, unlabeled, and val folders."""
    ensure_dirs()

    def _dir_has_images(directory: Path) -> bool:
        return any(directory.glob("*.jpg")) or any(directory.glob("*.jpeg")) or any(directory.glob("*.png"))

    split_dirs = [
        settings.TRAIN_DIR / "benign",
        settings.TRAIN_DIR / "malignant",
        settings.UNLABELED_DIR,
        settings.VAL_DIR / "benign",
        settings.VAL_DIR / "malignant",
    ]
    if any(_dir_has_images(directory) for directory in split_dirs):
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
    benign = len(list((train_dir / "benign").glob("*.jpg")))
    malignant = len(list((train_dir / "malignant").glob("*.jpg")))
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
        parent_dir = os.path.basename(os.path.dirname(image_filepath))
        label = 1.0 if parent_dir == "malignant" else 0.0

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label


class UnlabeledDataset(Dataset):
    """Dataset class for unlabeled data."""

    def __init__(
        self,
        root: str | Path,
        transform: Augm.Compose | None = None,
        weak_transform: Augm.Compose | None = None,
        strong_transform: Augm.Compose | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.samples = self._gather_unlabeled_samples(self.root)

    def _gather_unlabeled_samples(self, root: Path) -> list[tuple[Path, int]]:
        samples = []
        for filename in os.listdir(root):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = root / filename
                samples.append((path, -1))  # -1 indicates no label
        return samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        img = _read_rgb(path)
        if self.weak_transform is not None and self.strong_transform is not None:
            weak = self.weak_transform(image=img)["image"]
            strong = self.strong_transform(image=img)["image"]
            return weak, strong
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, target

    def __len__(self) -> int:
        return len(self.samples)


def get_transforms():
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

    return normalization, geo_transform, col_transform, pca_transform, weak_transform, strong_transform


def get_dataloaders(batch_size: int = 32):
    """Prepare dataloaders for training, unlabeled, and validation sets."""
    normalization, geo_t, col_t, pca_t, weak_t, strong_t = get_transforms()

    # Paths
    benign_dir = settings.TRAIN_DIR / "benign"
    malignant_dir = settings.TRAIN_DIR / "malignant"

    # Get filepaths for upsampling minority class (malignant)
    benign_filepaths = sorted(benign_dir.glob("*.jpg"))
    malignant_filepaths = sorted(malignant_dir.glob("*.jpg"))

    # Upsampling (Malignant x2 as in notebook)
    aug_train_filepaths = [*benign_filepaths, *malignant_filepaths]
    if not settings.USE_WEIGHTED_SAMPLER:
        aug_train_filepaths = [*aug_train_filepaths, *malignant_filepaths]
    random.seed(settings.SPLIT_SEED)
    random.shuffle(aug_train_filepaths)

    # Create augmented datasets
    geo_ds = AugmentedDataset(aug_train_filepaths, transform=geo_t)
    col_ds = AugmentedDataset(aug_train_filepaths, transform=col_t)
    pca_ds = AugmentedDataset(aug_train_filepaths, transform=pca_t)
    aug_train_dataset = ConcatDataset([geo_ds, col_ds, pca_ds])

    standard_train_filepaths = [*benign_filepaths, *malignant_filepaths]
    standard_train_dataset = AugmentedDataset(standard_train_filepaths, transform=normalization)
    train_dataset = ConcatDataset([standard_train_dataset, aug_train_dataset])

    if settings.SSL_METHOD == "fixmatch":
        unlabeled_dataset = UnlabeledDataset(settings.UNLABELED_DIR, weak_transform=weak_t, strong_transform=strong_t)
    else:
        unlabeled_dataset = UnlabeledDataset(settings.UNLABELED_DIR, transform=normalization)
    val_filepaths = sorted((settings.VAL_DIR / "benign").glob("*.jpg")) + sorted((settings.VAL_DIR / "malignant").glob("*.jpg"))
    val_dataset = AugmentedDataset(val_filepaths, transform=normalization)

    generator = torch.Generator().manual_seed(settings.SPLIT_SEED)
    sampler = None
    if settings.USE_WEIGHTED_SAMPLER:
        label_weights = []
        benign_weight = 1.0 / max(len(benign_filepaths), 1)
        malignant_weight = 1.0 / max(len(malignant_filepaths), 1)
        for dataset in train_dataset.datasets:
            if isinstance(dataset, AugmentedDataset):
                for filepath in dataset.images_filepaths:
                    parent_dir = Path(filepath).parent.name
                    label_weights.append(malignant_weight if parent_dir == "malignant" else benign_weight)
        sampler = WeightedRandomSampler(label_weights, num_samples=len(label_weights), replacement=True)

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
