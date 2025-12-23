import gc
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import settings
from .constants import DA_PROB_MAX, DA_PROB_MIN, EPSILON, EPSILON_8, RAMPUP_EXPONENT


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters()}
        self.backup: dict[str, torch.Tensor] = {}

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
        self.backup = {}


def _rampup_weight(current_epoch: int, rampup_epochs: int) -> float:
    """Compute rampup weight for unsupervised loss."""
    if rampup_epochs <= 0:
        return 1.0
    epoch = min(float(current_epoch), float(rampup_epochs))
    phase = 1.0 - epoch / float(rampup_epochs)
    return float(np.exp(-RAMPUP_EXPONENT * phase * phase))


def _apply_distribution_alignment(
    probs: torch.Tensor,
    ema_pos: torch.Tensor | None,
    target_pos: float,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply distribution alignment to match target class distribution."""
    batch_pos = probs.mean().clamp(DA_PROB_MIN, DA_PROB_MAX)
    ema_pos = batch_pos if ema_pos is None else momentum * ema_pos + (1 - momentum) * batch_pos
    target_pos_tensor = torch.tensor(target_pos, device=probs.device)
    scale_pos = target_pos_tensor / ema_pos
    scale_neg = (1 - target_pos_tensor) / (1 - ema_pos)
    adjusted = probs * scale_pos
    aligned = adjusted / (adjusted + (1 - probs) * scale_neg)
    return aligned, ema_pos


def _sharpen_probs(probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 1.0:
        return probs
    sharpened = probs.pow(1.0 / temperature)
    return sharpened / (sharpened + (1 - probs).pow(1.0 / temperature))


def _class_ratio_thresholds(
    pseudo_labels: torch.Tensor,
    base_tau: float,
    min_tau: float,
    pos_ratio: float,
    neg_ratio: float,
) -> torch.Tensor:
    max_ratio = max(pos_ratio, neg_ratio)
    tau_pos = max(min_tau, base_tau * (pos_ratio / max_ratio))
    tau_neg = max(min_tau, base_tau * (neg_ratio / max_ratio))
    return torch.where(
        pseudo_labels > 0.5,
        torch.tensor(tau_pos, device=pseudo_labels.device),
        torch.tensor(tau_neg, device=pseudo_labels.device),
    )


def _asymmetric_thresholds(pseudo_labels: torch.Tensor, base_tau: float) -> torch.Tensor:
    scale = 1.0 if settings.FIXMATCH_TAU <= 0 else base_tau / settings.FIXMATCH_TAU
    return torch.where(
        pseudo_labels > 0.5,
        torch.tensor(settings.FIXMATCH_TAU_POS * scale, device=pseudo_labels.device),
        torch.tensor(settings.FIXMATCH_TAU_NEG * scale, device=pseudo_labels.device),
    )


def _topk_mask(confidence: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(confidence)
    pos_conf = confidence[pseudo_labels > 0.5]
    neg_conf = confidence[pseudo_labels <= 0.5]
    if pos_conf.numel() > 0:
        k_pos = min(settings.FIXMATCH_TOPK_POS, pos_conf.numel())
        pos_thresh = torch.topk(pos_conf, k=k_pos, largest=True).values.min()
        mask = mask + ((confidence >= pos_thresh) & (pseudo_labels > 0.5)).float()
    if neg_conf.numel() > 0:
        k_neg = min(settings.FIXMATCH_TOPK_NEG, neg_conf.numel())
        neg_thresh = torch.topk(neg_conf, k=k_neg, largest=True).values.min()
        mask = mask + ((confidence >= neg_thresh) & (pseudo_labels <= 0.5)).float()
    return mask.clamp(max=1.0)


def _mixup_data(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply MixUp augmentation to create synthetic samples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def _mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for MixUp samples."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def _flexmatch_thresholds(
    confidence: torch.Tensor,
    pseudo_labels: torch.Tensor,
    ema_select: torch.Tensor,
    ema_total: torch.Tensor,
    base_tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute FlexMatch adaptive thresholds based on class selection ratios."""
    batch_pos = (pseudo_labels > 0.5).float()
    batch_neg = 1.0 - batch_pos
    ema_total = settings.FLEXMATCH_MOMENTUM * ema_total + (1 - settings.FLEXMATCH_MOMENTUM) * torch.stack([batch_neg.mean(), batch_pos.mean()])
    select_mask = (confidence >= base_tau).float()
    ema_select = settings.FLEXMATCH_MOMENTUM * ema_select + (1 - settings.FLEXMATCH_MOMENTUM) * torch.stack([(select_mask * batch_neg).mean(), (select_mask * batch_pos).mean()])
    class_ratio = ema_select / ema_total
    max_ratio = class_ratio.max().clamp(min=EPSILON)
    tau_neg = max(settings.FLEXMATCH_TAU_MIN, float(base_tau * (class_ratio[0] / max_ratio)))
    tau_pos = max(settings.FLEXMATCH_TAU_MIN, float(base_tau * (class_ratio[1] / max_ratio)))
    thresholds = torch.where(
        pseudo_labels > 0.5,
        torch.tensor(tau_pos, device=confidence.device),
        torch.tensor(tau_neg, device=confidence.device),
    )
    return thresholds, ema_select, ema_total


def _compute_supervised_loss(
    model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute supervised loss with optional MixUp augmentation."""
    use_mixup = settings.MIXUP_ENABLE and np.random.rand() < settings.MIXUP_PROB
    if not use_mixup:
        labeled_logits = model(images)
        supervised_loss = criterion(labeled_logits, targets)
        labeled_probs = torch.sigmoid(labeled_logits)
        return supervised_loss, labeled_probs

    mixed_imgs, labels_a, labels_b, lam = _mixup_data(images, targets, settings.MIXUP_ALPHA)
    labeled_logits = model(mixed_imgs)
    supervised_loss = _mixup_criterion(criterion, labeled_logits, labels_a, labels_b, lam)
    with torch.no_grad():
        orig_logits = model(images)
        labeled_probs = torch.sigmoid(orig_logits)
    return supervised_loss, labeled_probs


def _get_base_threshold(epoch: int) -> float:
    """Get base threshold with optional scheduling."""
    if not settings.FIXMATCH_TAU_SCHEDULE:
        return settings.FIXMATCH_TAU
    progress = min(epoch / max(1, settings.FIXMATCH_TAU_SCHEDULE_EPOCHS), 1.0)
    return settings.FIXMATCH_TAU_START + (settings.FIXMATCH_TAU_END - settings.FIXMATCH_TAU_START) * progress


def _get_thresholds(
    confidence: torch.Tensor,
    pseudo_labels: torch.Tensor,
    base_tau: float,
    epoch: int,
    ema_select: torch.Tensor,
    ema_total: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get thresholds based on configured strategy."""
    use_flexmatch = settings.FLEXMATCH_ENABLE and epoch >= settings.FLEXMATCH_WARMUP_EPOCHS
    if use_flexmatch:
        return _flexmatch_thresholds(confidence, pseudo_labels, ema_select, ema_total, base_tau)

    if settings.FIXMATCH_USE_ASYMMETRIC_TAU:
        thresholds = _asymmetric_thresholds(pseudo_labels, base_tau)
        return thresholds, ema_select, ema_total

    if settings.FIXMATCH_USE_CLASS_THRESHOLDS and settings.TRAIN_POS_RATIO is not None and settings.TRAIN_NEG_RATIO is not None:
        thresholds = _class_ratio_thresholds(
            pseudo_labels,
            base_tau,
            settings.FIXMATCH_MIN_TAU,
            settings.TRAIN_POS_RATIO,
            settings.TRAIN_NEG_RATIO,
        )
        return thresholds, ema_select, ema_total

    thresholds = torch.full_like(confidence, base_tau)
    return thresholds, ema_select, ema_total


def _compute_unsupervised_loss(
    model: nn.Module,
    weak_imgs: torch.Tensor,
    strong_imgs: torch.Tensor,
    epoch: int,
    ema_pos: torch.Tensor | None,
    ema_select: torch.Tensor,
    ema_total: torch.Tensor,
) -> tuple[torch.Tensor, float, float, float, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Compute unsupervised loss using FixMatch-style pseudo-labeling."""
    with torch.no_grad():
        weak_logits = model(weak_imgs)
        weak_probs = torch.sigmoid(weak_logits)

    if settings.FIXMATCH_DISTRIBUTION_ALIGNMENT and settings.TRAIN_POS_RATIO is not None:
        weak_probs, ema_pos = _apply_distribution_alignment(
            weak_probs,
            ema_pos,
            settings.TRAIN_POS_RATIO,
            settings.FIXMATCH_DA_MOMENTUM,
        )

    weak_probs = _sharpen_probs(weak_probs, settings.FIXMATCH_SHARPEN_T)
    confidence = torch.maximum(weak_probs, 1 - weak_probs)
    pseudo_labels = (weak_probs >= 0.5).float()

    base_tau = _get_base_threshold(epoch)
    thresholds, ema_select, ema_total = _get_thresholds(confidence, pseudo_labels, base_tau, epoch, ema_select, ema_total)

    strong_logits = model(strong_imgs)
    unsup_loss_all = F.binary_cross_entropy_with_logits(strong_logits, pseudo_labels, reduction="none")
    mask = _topk_mask(confidence, pseudo_labels) if settings.FIXMATCH_USE_TOPK else (confidence >= thresholds).float()

    if settings.SOFT_PSEUDO_LABELS:
        denom = (1 - thresholds).clamp_min(EPSILON)
        weights = ((confidence - thresholds) / denom).clamp(0, 1)
        unsup_loss = (unsup_loss_all * weights).sum() / (weights.sum() + EPSILON_8)
    else:
        unsup_loss = (unsup_loss_all * mask).sum() / (mask.sum() + EPSILON_8)

    accepted_total = float(mask.sum().item())
    accepted_pos = float((mask * pseudo_labels).sum().item())
    accepted_neg = float((mask * (1 - pseudo_labels)).sum().item())

    return (
        unsup_loss,
        accepted_total,
        accepted_pos,
        accepted_neg,
        ema_pos,
        ema_select,
        ema_total,
    )


def train_one_epoch_fixmatch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: Optimizer,
    supervised_criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: torch.amp.GradScaler | None = None,
    ema: EMA | None = None,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Train using FixMatch-style pseudo-labeling."""
    model.train()
    total_loss_accum = 0.0
    sup_loss_accum = 0.0
    unsup_loss_accum = 0.0
    num_batches = 0
    labeled_preds_list = []
    labeled_labels_list = []
    ema_pos = None
    ema_select = torch.tensor([0.0, 0.0], device=device)
    ema_total = torch.tensor([EPSILON, EPSILON], device=device)

    unlabeled_iter = itertools.cycle(unlabeled_loader)
    accepted_pos = 0.0
    accepted_neg = 0.0
    accepted_total = 0.0

    # We use labeled_loader to drive the loop to ensure we see all labeled data
    for labeled_imgs, labeled_targets in labeled_loader:
        try:
            weak_imgs, strong_imgs = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = itertools.cycle(unlabeled_loader)
            weak_imgs, strong_imgs = next(unlabeled_iter)

        num_batches += 1

        labeled_imgs = labeled_imgs.to(device)
        labeled_targets = labeled_targets.float().unsqueeze(1).to(device)
        weak_imgs = weak_imgs.to(device)
        strong_imgs = strong_imgs.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type="cuda", enabled=settings.USE_AMP and device.type == "cuda"):
            supervised_loss, labeled_probs = _compute_supervised_loss(model, supervised_criterion, labeled_imgs, labeled_targets)
            labeled_preds_list.extend(labeled_probs.detach().cpu().numpy())
            labeled_labels_list.extend(labeled_targets.detach().cpu().numpy())

            unsup_loss, acc_total, acc_pos, acc_neg, ema_pos, ema_select, ema_total = _compute_unsupervised_loss(
                model, weak_imgs, strong_imgs, epoch, ema_pos, ema_select, ema_total
            )
            accepted_total += acc_total
            accepted_pos += acc_pos
            accepted_neg += acc_neg

            lambda_u = settings.FIXMATCH_LAMBDA_U * _rampup_weight(epoch, settings.FIXMATCH_RAMPUP_EPOCHS)
            total_loss = supervised_loss + lambda_u * unsup_loss

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.MAX_GRAD_NORM)
            optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss_accum += total_loss.item()
        sup_loss_accum += supervised_loss.item()
        unsup_loss_accum += unsup_loss.item()

    if num_batches == 0:
        raise RuntimeError("No batches processed; check your data loaders.")

    avg_loss = total_loss_accum / num_batches
    avg_sup_loss = sup_loss_accum / num_batches
    avg_unsup_loss = unsup_loss_accum / num_batches

    labeled_labels = np.asarray(labeled_labels_list).reshape(-1)
    labeled_preds = np.asarray(labeled_preds_list).reshape(-1)
    labeled_hard_preds = (labeled_preds >= 0.5).astype(np.float32)
    acc = accuracy_score(labeled_labels, labeled_hard_preds)
    has_multiple_classes = len(np.unique(labeled_labels)) > 1
    auc = roc_auc_score(labeled_labels, labeled_preds) if has_multiple_classes else 0.0
    return (
        avg_loss,
        avg_sup_loss,
        avg_unsup_loss,
        acc,
        auc,
        accepted_total,
        accepted_pos,
        accepted_neg,
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ema: EMA | None = None,
) -> tuple[float, float, float, float, float]:
    """Validate the student model."""
    model.eval()
    if ema is not None:
        ema.apply(model)
    val_labels_list = []
    val_preds_list = []

    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        logits = model(imgs)
        preds = torch.sigmoid(logits)

        val_preds_list.extend(preds.cpu().numpy())
        val_labels_list.extend(targets.cpu().numpy())

    val_labels = np.asarray(val_labels_list).reshape(-1)
    val_preds = np.asarray(val_preds_list).reshape(-1)
    val_hard_preds = (val_preds >= 0.5).astype(np.float32)

    has_multiple_classes = len(np.unique(val_labels)) > 1
    acc = accuracy_score(val_labels, val_hard_preds)
    auc = roc_auc_score(val_labels, val_preds) if has_multiple_classes else 0.0
    ap = average_precision_score(val_labels, val_preds) if has_multiple_classes else 0.0

    pos_mask = val_labels > 0.5
    sensitivity = float(np.mean(val_hard_preds[pos_mask])) if np.any(pos_mask) else 0.0
    pos_rate = float(np.mean(val_hard_preds)) if len(val_hard_preds) > 0 else 0.0

    if ema is not None:
        ema.restore(model)

    return acc, auc, ap, pos_rate, sensitivity


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    supervised_criterion: nn.Module,
    device: str,
    epochs: int,
):
    """Main training loop."""
    device = torch.device(device)
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=settings.USE_AMP and device.type == "cuda")
    ema = EMA(model, settings.EMA_DECAY) if settings.EMA_ENABLE else None

    history = {
        "loss_total": [],
        "loss_sup": [],
        "loss_unsup": [],
        "train_acc": [],
        "train_auc": [],
        "val_acc": [],
        "val_auc": [],
        "val_ap": [],
    }

    def _select_metric(name: str, *, val_auc: float, val_ap: float) -> float:
        if name == "val_auc":
            return val_auc
        if name == "val_ap":
            return val_ap
        raise ValueError(f"Unsupported metric '{name}'. Use 'val_auc' or 'val_ap'.")

    best_metric_name = settings.BEST_METRIC
    early_stop_metric_name = settings.EARLY_STOP_METRIC or best_metric_name

    best_metric = -1.0
    best_state = None
    best_early_stop_metric = -1.0
    best_early_stop_epoch = 0
    patience_left = settings.EARLY_STOP_PATIENCE

    for epoch in range(epochs):
        if epoch < settings.FREEZE_BACKBONE_EPOCHS:
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif epoch == settings.FREEZE_BACKBONE_EPOCHS:
            for param in model.encoder.parameters():
                param.requires_grad = True
        (
            avg_loss,
            sup_loss,
            unsup_loss,
            train_acc,
            train_auc,
            accepted_total,
            accepted_pos,
            _,
        ) = train_one_epoch_fixmatch(
            model,
            train_loader,
            unlabeled_loader,
            optimizer,
            supervised_criterion,
            device,
            epoch,
            scaler,
            ema,
        )

        val_acc, val_auc, val_ap, val_pos_rate, val_sens = validate(model, val_loader, device, ema)
        scheduler.step()

        # Update history
        history["loss_total"].append(avg_loss)
        history["loss_sup"].append(sup_loss)
        history["loss_unsup"].append(unsup_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        accepted_ratio = accepted_pos / accepted_total if accepted_total > 0 else 0.0
        print(
            f"Epoch [{epoch + 1:02}/{epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc * 100:.2f}% "
            f"- Val Acc: {val_acc * 100:.2f}% - Val AUC: {val_auc:.4f} - Val AP: {val_ap:.4f} "
            f"- Val Pos%: {val_pos_rate * 100:.1f}% - Val Sens: {val_sens * 100:.1f}% - Accepted Pos%: {accepted_ratio * 100:.1f}%"
        )

        early_stop_value = _select_metric(early_stop_metric_name, val_auc=val_auc, val_ap=val_ap)
        if early_stop_value > best_early_stop_metric + settings.EARLY_STOP_MIN_DELTA:
            best_early_stop_metric = early_stop_value
            best_early_stop_epoch = epoch + 1
            patience_left = settings.EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    "Early stopping triggered "
                    f"(best {early_stop_metric_name}={best_early_stop_metric:.4f} at epoch {best_early_stop_epoch})."
                )
                break

        metric_value = _select_metric(best_metric_name, val_auc=val_auc, val_ap=val_ap)
        if metric_value <= best_metric:
            continue

        best_metric = metric_value
        if ema is not None:
            best_state = {}
            for name, tensor in model.state_dict().items():
                if name in ema.shadow:
                    best_state[name] = ema.shadow[name].detach().cpu().clone()
                else:
                    best_state[name] = tensor.detach().cpu().clone()
        else:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if settings.SAVE_BEST_CHECKPOINT:
            checkpoint_dir = Path(settings.CHECKPOINT_DIR)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": best_state,
                    "val_auc": val_auc,
                    "val_ap": val_ap,
                    "val_acc": val_acc,
                    "best_metric": best_metric_name,
                    "ema_enabled": settings.EMA_ENABLE,
                },
                checkpoint_dir / "best.pt",
            )

        # Cleanup
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, history
