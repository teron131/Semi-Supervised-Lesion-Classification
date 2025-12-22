import math

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import settings
from .data import get_class_counts, get_dataloaders, prepare_data
from .losses import BCEFocalLoss
from .models import ClassifierModel, get_convnext_tiny
from .trainer import run_training
from .utils import plot_history, save_history, set_seed


def _update_class_ratios() -> tuple[int, int]:
    benign_count, malignant_count = get_class_counts(settings.TRAIN_DIR)
    total_count = benign_count + malignant_count
    if total_count > 0:
        settings.TRAIN_NEG_RATIO = benign_count / total_count
        settings.TRAIN_POS_RATIO = malignant_count / total_count
        print(f"Train split: benign={benign_count} malignant={malignant_count}")

    val_benign, val_malignant = get_class_counts(settings.VAL_DIR)
    val_total = val_benign + val_malignant
    if val_total > 0:
        print(f"Val split: benign={val_benign} malignant={val_malignant}")
    return benign_count, malignant_count


def _build_supervised_loss(benign_count: int, malignant_count: int) -> nn.Module:
    if settings.SUPERVISED_LOSS == "bce":
        pos_weight_value = settings.POS_WEIGHT
        if settings.AUTO_POS_WEIGHT and benign_count > 0 and malignant_count > 0:
            pos_weight_value = min(benign_count / malignant_count, settings.POS_WEIGHT_MAX)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=settings.DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return BCEFocalLoss(gamma=settings.FOCAL_GAMMA, alpha=settings.FOCAL_ALPHA)


def main():
    # 1. Setup
    set_seed(42)

    # 2. Data Preparation (if needed)
    prepare_data(settings.DATA_DIR)

    benign_count, malignant_count = _update_class_ratios()

    # 3. Data Loaders
    train_loader, unlabeled_loader, val_loader = get_dataloaders(settings.BATCH_SIZE)

    # 4. Model
    backbone, feature_dim = get_convnext_tiny(pre_trained=settings.PRE_TRAINED)
    model = ClassifierModel(backbone, feature_dim, settings.NUM_CLASSES)

    # 5. Optimizer, Loss, Scheduler
    optimizer = AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    supervised_criterion = _build_supervised_loss(benign_count, malignant_count)

    def lr_lambda(epoch: int) -> float:
        if epoch < settings.WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, settings.WARMUP_EPOCHS))
        progress = (epoch - settings.WARMUP_EPOCHS) / float(max(1, settings.EPOCHS - settings.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    # 6. Training
    model, history = run_training(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        supervised_criterion=supervised_criterion,
        device=settings.DEVICE,
        epochs=settings.EPOCHS,
    )

    # 7. Visualization
    save_history(history, settings.RESULTS_DIR)
    plot_history(history, settings.RESULTS_DIR)


if __name__ == "__main__":
    main()
