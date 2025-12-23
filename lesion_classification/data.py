from dataclasses import dataclass
from pathlib import Path
import shutil

import albumentations as Augm
from albumentations.augmentations import ColorJitter, FancyPCA
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import ensure_dirs, settings
from .constants import (
    CLASS_BENIGN,
    CLASS_MALIGNANT,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_BENIGN,
    LABEL_MALIGNANT,
)


@dataclass(frozen=True)
class DataTransforms:
    normalize: Augm.Compose
    geo: Augm.Compose
    color: Augm.Compose
    pca: Augm.Compose
    weak: Augm.Compose
    strong: Augm.Compose
    train_labeled: Augm.Compose


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
    return LABEL_MALIGNANT if parent_dir == CLASS_MALIGNANT else LABEL_BENIGN


def _copy_images(image_indices: np.ndarray, image_ids: list[str], labels: list[str] | None, src_dir: Path, dst_dir: Path):
    """Copy images from source to destination directory."""
    for i in image_indices:
        name = image_ids[i]
        src_path = src_dir / f"{name}.jpg"
        if src_path.exists():
            if labels is not None:
                label = labels[i]
                (dst_dir / label).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_dir / label / f"{name}.jpg")
            else:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_dir / f"{name}.jpg")


def prepare_data(data_root: Path):
    """Split raw ISIC data into train, unlabeled, and val folders."""
    ensure_dirs()

    split_dirs = [
        settings.TRAIN_DIR / CLASS_BENIGN,
        settings.TRAIN_DIR / CLASS_MALIGNANT,
        settings.UNLABELED_DIR,
        settings.VAL_DIR / CLASS_BENIGN,
        settings.VAL_DIR / CLASS_MALIGNANT,
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

    _copy_images(ix_train, img_lis, lbl_lis, src_data_dir, settings.TRAIN_DIR)
    _copy_images(ix_unlabeled, img_lis, None, src_data_dir, settings.UNLABELED_DIR)
    _copy_images(ix_val, img_lis, lbl_lis, src_data_dir, settings.VAL_DIR)

    print(f"Data split completed: {len(ix_train)} train, {len(ix_unlabeled)} unlabeled, {len(ix_val)} val.")


def get_class_counts(train_dir: Path) -> tuple[int, int]:
    benign = len(_list_images(train_dir / CLASS_BENIGN))
    malignant = len(_list_images(train_dir / CLASS_MALIGNANT))
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


def _base_transform() -> list:
    """Base resize and normalization pipeline."""
    return [
        Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
        Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def _normalization_transform() -> Augm.Compose:
    """Simple normalization transform for validation."""
    return Augm.Compose(
        [
            Augm.Resize(settings.IMAGE_RESIZE, settings.IMAGE_RESIZE),
            Augm.CenterCrop(settings.IMAGE_SIZE, settings.IMAGE_SIZE),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _geometric_transform() -> Augm.Compose:
    """Geometric augmentation transform."""
    return Augm.Compose(
        [
            *_base_transform()[:-1],  # Resize and normalize, skip ToTensorV2
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=0.5),
            Augm.VerticalFlip(p=0.5),
            Augm.Rotate(limit=30, p=0.5),
            Augm.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.3),
            Augm.RandomBrightnessContrast(p=0.2),
            ToTensorV2(),
        ]
    )


def _color_transform() -> Augm.Compose:
    """Color augmentation transform."""
    return Augm.Compose(
        [
            *_base_transform()[:-1],
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=0.5),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            Augm.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            Augm.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
            ToTensorV2(),
        ]
    )


def _pca_transform() -> Augm.Compose:
    """PCA-based color augmentation transform."""
    return Augm.Compose(
        [
            *_base_transform()[:-1],
            Augm.RandomCrop(width=settings.IMAGE_SIZE, height=settings.IMAGE_SIZE),
            Augm.HorizontalFlip(p=0.5),
            Augm.VerticalFlip(p=0.3),
            FancyPCA(alpha=0.1),
            Augm.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            ToTensorV2(),
        ]
    )


def _weak_transform() -> Augm.Compose:
    """Weak augmentation for FixMatch (labeled and unlabeled weak views)."""
    return Augm.Compose(
        [
            Augm.RandomResizedCrop(size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE), scale=(0.8, 1.0)),
            Augm.HorizontalFlip(p=0.5),
            Augm.VerticalFlip(p=0.5),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _get_strong_augment() -> list:
    """Get strong augmentation component based on available albumentations version."""
    if hasattr(Augm, "RandAugment"):
        return [Augm.RandAugment(num_ops=2, magnitude=7)]
    if hasattr(Augm, "TrivialAugmentWide"):
        return [Augm.TrivialAugmentWide()]
    return [
        Augm.OneOf(
            [
                ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                Augm.GaussianBlur(blur_limit=(3, 5)),
            ],
            p=0.8,
        )
    ]


def _strong_transform() -> Augm.Compose:
    """Strong augmentation for FixMatch unlabeled data."""
    strong_augment = _get_strong_augment()
    return Augm.Compose(
        [
            Augm.RandomResizedCrop(size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE), scale=(0.5, 1.0)),
            Augm.HorizontalFlip(p=0.5),
            Augm.VerticalFlip(p=0.5),
            Augm.Rotate(limit=45, p=0.5),
            *strong_augment,
            Augm.OneOf(
                [
                    Augm.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                    Augm.ElasticTransform(alpha=1, sigma=50, p=1.0),
                    Augm.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-20, 20), p=1.0),
                ],
                p=0.3,
            ),
            Augm.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 32), hole_width_range=(1, 32), p=0.5),
            Augm.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _labeled_transform() -> Augm.Compose:
    """Medium augmentation for labeled data to reduce overfitting."""
    return Augm.Compose(
        [
            Augm.RandomResizedCrop(size=(settings.IMAGE_SIZE, settings.IMAGE_SIZE), scale=(0.7, 1.0)),
            Augm.HorizontalFlip(p=0.5),
            Augm.VerticalFlip(p=0.5),
            Augm.Rotate(limit=30, p=0.5),
            Augm.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            Augm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_transforms() -> DataTransforms:
    """Define all data transformations."""
    weak = _weak_transform()
    return DataTransforms(
        normalize=_normalization_transform(),
        geo=_geometric_transform(),
        color=_color_transform(),
        pca=_pca_transform(),
        weak=weak,
        strong=_strong_transform(),
        train_labeled=_labeled_transform(),
    )


def _build_weighted_sampler(train_dataset: Dataset, benign_count: int, malignant_count: int) -> WeightedRandomSampler:
    label_weights = []
    benign_weight = 1.0 / max(benign_count, 1)
    malignant_weight = 1.0 / max(malignant_count, 1)

    if isinstance(train_dataset, AugmentedDataset):
        for filepath in train_dataset.images_filepaths:
            weight = malignant_weight if _label_from_path(filepath) == LABEL_MALIGNANT else benign_weight
            label_weights.append(weight)
    else:
        for _, label in train_dataset:
            weight = malignant_weight if label == LABEL_MALIGNANT else benign_weight
            label_weights.append(weight)

    num_samples = settings.TRAIN_STEPS_PER_EPOCH * settings.BATCH_SIZE
    return WeightedRandomSampler(label_weights, num_samples=num_samples, replacement=True)


def get_dataloaders(batch_size: int = 32):
    """Prepare dataloaders for training, unlabeled, and validation sets."""
    transforms = get_transforms()

    benign_dir = settings.TRAIN_DIR / CLASS_BENIGN
    malignant_dir = settings.TRAIN_DIR / CLASS_MALIGNANT

    benign_filepaths = _list_images(benign_dir)
    malignant_filepaths = _list_images(malignant_dir)
    train_filepaths = [*benign_filepaths, *malignant_filepaths]
    train_dataset = AugmentedDataset(train_filepaths, transform=transforms.train_labeled)

    unlabeled_dataset = UnlabeledDataset(settings.UNLABELED_DIR, weak_transform=transforms.weak, strong_transform=transforms.strong)
    val_filepaths = _list_images(settings.VAL_DIR / CLASS_BENIGN) + _list_images(settings.VAL_DIR / CLASS_MALIGNANT)
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
        batch_size=batch_size * settings.FIXMATCH_MU,
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
