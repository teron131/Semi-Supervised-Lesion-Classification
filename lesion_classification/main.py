import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import settings
from .data import get_class_counts, get_dataloaders, prepare_data
from .losses import BCEFocalLoss
from .models import MeanTeacherModel, ResnetModel, get_resnet50
from .trainer import run_training
from .utils import plot_history, save_history, set_seed


def main():
    # 1. Setup
    set_seed(42)

    # 2. Data Preparation (if needed)
    prepare_data(settings.DATA_DIR)

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

    # 3. Data Loaders
    train_loader, unlabeled_loader, val_loader = get_dataloaders(settings.BATCH_SIZE)

    # 4. Model
    resnet50 = get_resnet50(pre_trained=settings.PRE_TRAINED, dropout=settings.DROPOUT)
    base_model = ResnetModel(resnet50, settings.NUM_CLASSES)
    if settings.SSL_METHOD == "fixmatch":
        model = base_model
    else:
        model = MeanTeacherModel(
            base_model,
            ema_decay=settings.EMA_DECAY,
            consistency_max_weight=settings.CONSISTENCY_MAX_WEIGHT,
            rampup_epochs=settings.CONSISTENCY_RAMPUP_EPOCHS,
        )

    # 5. Optimizer, Loss, Scheduler
    optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
    if settings.SUPERVISED_LOSS == "bce":
        pos_weight_value = settings.POS_WEIGHT
        if settings.AUTO_POS_WEIGHT and benign_count > 0 and malignant_count > 0:
            pos_weight_value = benign_count / malignant_count
            pos_weight_value = min(pos_weight_value, settings.POS_WEIGHT_MAX)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=settings.DEVICE)
        supervised_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        supervised_criterion = BCEFocalLoss(gamma=settings.FOCAL_GAMMA, alpha=settings.FOCAL_ALPHA)
    consistency_criterion = nn.MSELoss()

    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=settings.EPOCHS)

    # 6. Training
    model, history = run_training(
        model=model,
        train_loader=train_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        supervised_criterion=supervised_criterion,
        consistency_criterion=consistency_criterion,
        device=settings.DEVICE,
        epochs=settings.EPOCHS,
    )

    # 7. Visualization
    save_history(history, settings.RESULTS_DIR)
    plot_history(history, settings.RESULTS_DIR)


if __name__ == "__main__":
    main()
