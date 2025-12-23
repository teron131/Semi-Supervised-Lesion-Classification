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
    """Update class ratios in settings and print statistics."""
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
    """Build supervised loss function based on configuration."""
    if settings.SUPERVISED_LOSS == "bce":
        pos_weight_value = settings.POS_WEIGHT
        if settings.AUTO_POS_WEIGHT and benign_count > 0 and malignant_count > 0:
            pos_weight_value = min(benign_count / malignant_count, settings.POS_WEIGHT_MAX)
        pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=settings.DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal_alpha = settings.FOCAL_ALPHA
    if settings.AUTO_FOCAL_ALPHA and benign_count > 0 and malignant_count > 0:
        pos_weight_value = min(benign_count / malignant_count, settings.POS_WEIGHT_MAX)
        focal_alpha = float(pos_weight_value / (1.0 + pos_weight_value))
    return BCEFocalLoss(gamma=settings.FOCAL_GAMMA, alpha=focal_alpha)


def _build_optimizer(model: ClassifierModel) -> AdamW:
    """Build optimizer with layer-wise learning rate decay."""
    base_lr = settings.LEARNING_RATE
    decay = settings.LR_LAYER_DECAY
    param_groups = []
    added: set[int] = set()

    def split_decay(params: list[nn.Parameter], names: list[str]) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []
        for param, name in zip(params, names, strict=True):
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return decay_params, no_decay_params

    def add_group(params: list[nn.Parameter], lr: float, names: list[str]) -> None:
        decay_params, no_decay_params = split_decay(params, names)
        if decay_params:
            param_groups.append({"params": decay_params, "lr": lr, "weight_decay": settings.WEIGHT_DECAY})
            added.update(id(p) for p in decay_params)
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "lr": lr, "weight_decay": 0.0})
            added.update(id(p) for p in no_decay_params)

    backbone = model.encoder[0] if isinstance(model.encoder, nn.Sequential) else model.encoder
    if hasattr(backbone, "features"):
        stages = list(backbone.features)
        n_stages = max(len(stages), 1)
        for idx, stage in enumerate(stages):
            lr = base_lr * (decay ** (n_stages - 1 - idx))
            stage_named_params = list(stage.named_parameters())
            if stage_named_params:
                stage_params = [param for _, param in stage_named_params]
                stage_names = [name for name, _ in stage_named_params]
                add_group(stage_params, lr, stage_names)

    head_lr = base_lr * settings.HEAD_LR_MULT
    head_named_params = list(model.classifier.named_parameters())
    if head_named_params:
        head_params = [param for _, param in head_named_params]
        head_names = [name for name, _ in head_named_params]
        add_group(head_params, head_lr, head_names)

    remaining_named = [(name, param) for name, param in model.named_parameters() if id(param) not in added]
    if remaining_named:
        remaining_params = [param for _, param in remaining_named]
        remaining_names = [name for name, _ in remaining_named]
        add_group(remaining_params, base_lr, remaining_names)

    return AdamW(param_groups, lr=base_lr, weight_decay=0.0)


def _build_scheduler(optimizer: AdamW) -> LambdaLR:
    """Build learning rate scheduler with warmup and cosine annealing."""

    def lr_lambda(epoch: int) -> float:
        if epoch < settings.WARMUP_EPOCHS:
            return float(epoch + 1) / float(max(1, settings.WARMUP_EPOCHS))
        progress = (epoch - settings.WARMUP_EPOCHS) / float(max(1, settings.EPOCHS - settings.WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def _initialize_model_bias(model: ClassifierModel, benign_count: int, malignant_count: int) -> None:
    """Initialize classifier bias from class prior."""
    if not settings.INIT_BIAS_FROM_PRIOR or malignant_count == 0 or benign_count == 0:
        return
    prior = malignant_count / (benign_count + malignant_count)
    bias_value = math.log(prior / (1 - prior))
    with torch.no_grad():
        model.classifier.bias.fill_(bias_value)


def main():
    """Main training entry point."""
    set_seed(42)
    prepare_data(settings.DATA_DIR)

    benign_count, malignant_count = _update_class_ratios()
    train_loader, unlabeled_loader, val_loader = get_dataloaders(settings.BATCH_SIZE)

    backbone, feature_dim = get_convnext_tiny(pre_trained=settings.PRE_TRAINED)
    model = ClassifierModel(backbone, feature_dim, settings.NUM_CLASSES)
    _initialize_model_bias(model, benign_count, malignant_count)

    optimizer = _build_optimizer(model)
    supervised_criterion = _build_supervised_loss(benign_count, malignant_count)
    scheduler = _build_scheduler(optimizer)

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

    save_history(history, settings.RESULTS_DIR)
    plot_history(history, settings.RESULTS_DIR)


if __name__ == "__main__":
    main()
