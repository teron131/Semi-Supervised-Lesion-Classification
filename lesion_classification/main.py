from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .config import settings
from .data import get_dataloaders, prepare_data
from .losses import BCEFocalLoss
from .models import MeanTeacherModel, ResnetModel, get_resnet50
from .trainer import run_training
from .utils import plot_history, set_seed


def main():
    # 1. Setup
    set_seed(42)

    # 2. Data Preparation (if needed)
    prepare_data(settings.DATA_DIR)

    # 3. Data Loaders
    train_loader, unlabeled_loader, val_loader = get_dataloaders(settings.BATCH_SIZE)

    # 4. Model
    resnet50 = get_resnet50(pre_trained=settings.PRE_TRAINED, dropout=settings.DROPOUT)
    base_model = ResnetModel(resnet50, settings.NUM_CLASSES)
    model = MeanTeacherModel(base_model, ema_decay=settings.EMA_DECAY)

    # 5. Optimizer, Loss, Scheduler
    optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=settings.WEIGHT_DECAY)
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
    plot_history(history)


if __name__ == "__main__":
    main()
