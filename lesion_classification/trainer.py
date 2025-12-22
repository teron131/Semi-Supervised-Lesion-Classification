import gc
import itertools

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from .config import settings


def train_one_epoch_mean_teacher(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: Optimizer,
    supervised_criterion: nn.Module,
    consistency_criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, float, float]:
    """Train the Mean Teacher model for one epoch."""
    model.student_model.train()
    model.teacher_model.eval()

    total_loss_accum = 0.0
    sup_loss_accum = 0.0
    con_loss_accum = 0.0
    num_batches = 0

    unlabeled_labels_list = []
    unlabeled_preds_list = []

    unlabeled_iter = iter(unlabeled_loader)
    labeled_iter = itertools.cycle(labeled_loader)

    for _ in range(len(unlabeled_loader)):
        labeled_imgs, labeled_targets = next(labeled_iter)
        unlabeled_imgs, _ = next(unlabeled_iter)
        num_batches += 1
        labeled_imgs = labeled_imgs.to(device)
        labeled_targets = labeled_targets.float().unsqueeze(1).to(device)
        unlabeled_imgs = unlabeled_imgs.to(device)

        optimizer.zero_grad()

        # 1. Supervised Loss (Student on Labeled Data)
        labeled_preds = model.student_model(labeled_imgs)
        supervised_loss = supervised_criterion(labeled_preds, labeled_targets)

        # 2. Consistency Loss (Student vs Teacher on Unlabeled Data)
        unlabeled_preds_student = model.student_model(unlabeled_imgs)
        with torch.no_grad():
            unlabeled_preds_teacher = model.teacher_model(unlabeled_imgs)

        consistency_weight = model.get_consistency_weight(epoch)
        teacher_logits = unlabeled_preds_teacher / settings.TEACHER_TEMPERATURE
        consistency_loss = consistency_weight * consistency_criterion(unlabeled_preds_student, teacher_logits)

        # 3. Total Loss and Optimization
        total_loss = supervised_loss + consistency_loss
        total_loss.backward()
        optimizer.step()

        # 4. Update Teacher Parameters (EMA)
        model.update_teacher_model(current_epoch=epoch)

        # Accumulate metrics
        total_loss_accum += total_loss.item()
        sup_loss_accum += supervised_loss.item()
        con_loss_accum += consistency_loss.item()

        unlabeled_probs_student = torch.sigmoid(unlabeled_preds_student)
        unlabeled_probs_teacher = torch.sigmoid(unlabeled_preds_teacher)
        unlabeled_preds_list.extend(unlabeled_probs_student.detach().cpu().numpy())
        unlabeled_labels_list.extend(np.around(unlabeled_probs_teacher.detach().cpu().numpy()))

    # Calculate metrics for unlabeled data (pseudo-accuracy/AUC)
    unlabeled_labels = np.array(unlabeled_labels_list)
    unlabeled_preds = np.array(unlabeled_preds_list)

    acc = accuracy_score(unlabeled_labels, np.round(unlabeled_preds))
    auc = roc_auc_score(unlabeled_labels, unlabeled_preds) if len(np.unique(unlabeled_labels)) > 1 else 0.0

    if num_batches == 0:
        raise RuntimeError("No batches processed; check your data loaders.")

    avg_loss = total_loss_accum / num_batches
    avg_sup_loss = sup_loss_accum / num_batches
    avg_con_loss = con_loss_accum / num_batches
    return avg_loss, avg_sup_loss, avg_con_loss, acc, auc


def train_one_epoch_fixmatch(
    model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    optimizer: Optimizer,
    supervised_criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, float, float]:
    """Train using FixMatch-style pseudo-labeling."""
    model.train()
    total_loss_accum = 0.0
    sup_loss_accum = 0.0
    unsup_loss_accum = 0.0
    num_batches = 0
    labeled_preds_list = []
    labeled_labels_list = []

    def rampup_weight(current_epoch: int) -> float:
        if settings.FIXMATCH_RAMPUP_EPOCHS <= 0:
            return 1.0
        epoch = min(float(current_epoch), float(settings.FIXMATCH_RAMPUP_EPOCHS))
        phase = 1.0 - epoch / float(settings.FIXMATCH_RAMPUP_EPOCHS)
        return float(np.exp(-5.0 * phase * phase))

    labeled_iter = itertools.cycle(labeled_loader)
    for weak_imgs, strong_imgs in unlabeled_loader:
        labeled_imgs, labeled_targets = next(labeled_iter)
        num_batches += 1

        labeled_imgs = labeled_imgs.to(device)
        labeled_targets = labeled_targets.float().unsqueeze(1).to(device)
        weak_imgs = weak_imgs.to(device)
        strong_imgs = strong_imgs.to(device)

        optimizer.zero_grad()

        labeled_logits = model(labeled_imgs)
        supervised_loss = supervised_criterion(labeled_logits, labeled_targets)
        labeled_probs = torch.sigmoid(labeled_logits)
        labeled_preds_list.extend(labeled_probs.detach().cpu().numpy())
        labeled_labels_list.extend(labeled_targets.detach().cpu().numpy())

        with torch.no_grad():
            weak_logits = model(weak_imgs)
            weak_probs = torch.sigmoid(weak_logits)
            confidence = torch.maximum(weak_probs, 1 - weak_probs)
            pseudo_labels = (weak_probs >= 0.5).float()
            if settings.FIXMATCH_USE_CLASS_THRESHOLDS and settings.TRAIN_POS_RATIO is not None and settings.TRAIN_NEG_RATIO is not None:
                max_ratio = max(settings.TRAIN_POS_RATIO, settings.TRAIN_NEG_RATIO)
                tau_pos = max(settings.FIXMATCH_MIN_TAU, settings.FIXMATCH_TAU * (settings.TRAIN_POS_RATIO / max_ratio))
                tau_neg = max(settings.FIXMATCH_MIN_TAU, settings.FIXMATCH_TAU * (settings.TRAIN_NEG_RATIO / max_ratio))
                thresholds = torch.where(pseudo_labels > 0.5, torch.tensor(tau_pos, device=confidence.device), torch.tensor(tau_neg, device=confidence.device))
                mask = (confidence >= thresholds).float()
            else:
                mask = (confidence >= settings.FIXMATCH_TAU).float()

        strong_logits = model(strong_imgs)
        unsup_loss = F.binary_cross_entropy_with_logits(strong_logits, pseudo_labels, reduction="none")
        unsup_loss = (unsup_loss * mask).sum() / (mask.sum() + 1e-8)

        lambda_u = settings.FIXMATCH_LAMBDA_U * rampup_weight(epoch)
        total_loss = supervised_loss + lambda_u * unsup_loss
        total_loss.backward()
        optimizer.step()

        total_loss_accum += total_loss.item()
        sup_loss_accum += supervised_loss.item()
        unsup_loss_accum += unsup_loss.item()

    if num_batches == 0:
        raise RuntimeError("No batches processed; check your data loaders.")

    avg_loss = total_loss_accum / num_batches
    avg_sup_loss = sup_loss_accum / num_batches
    avg_unsup_loss = unsup_loss_accum / num_batches

    labeled_labels = np.array(labeled_labels_list)
    labeled_preds = np.array(labeled_preds_list)
    acc = accuracy_score(labeled_labels, np.round(labeled_preds))
    auc = roc_auc_score(labeled_labels, labeled_preds) if len(np.unique(labeled_labels)) > 1 else 0.0
    return avg_loss, avg_sup_loss, avg_unsup_loss, acc, auc


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float, float]:
    """Validate the student model."""
    if hasattr(model, "student_model"):
        model.student_model.eval()
        eval_model = model.student_model
    else:
        model.eval()
        eval_model = model
    val_labels_list = []
    val_preds_list = []

    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        logits = eval_model(imgs)
        preds = torch.sigmoid(logits)

        val_preds_list.extend(preds.cpu().numpy())
        val_labels_list.extend(targets.cpu().numpy())

    val_labels = np.array(val_labels_list)
    val_preds = np.array(val_preds_list)

    acc = accuracy_score(val_labels, np.round(val_preds))
    auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.0
    ap = average_precision_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.0

    return acc, auc, ap


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    supervised_criterion: nn.Module,
    consistency_criterion: nn.Module,
    device: str,
    epochs: int,
):
    """Main training loop."""
    device = torch.device(device)
    model.to(device)

    history = {
        "loss_total": [],
        "loss_sup": [],
        "loss_con": [],
        "train_acc": [],
        "train_auc": [],
        "val_acc": [],
        "val_auc": [],
        "val_ap": [],
    }

    best_val_auc = -1.0
    best_state = None
    patience_left = settings.EARLY_STOP_PATIENCE

    for epoch in range(epochs):
        if settings.SSL_METHOD == "fixmatch":
            avg_loss, sup_loss, con_loss, train_acc, train_auc = train_one_epoch_fixmatch(model, train_loader, unlabeled_loader, optimizer, supervised_criterion, device, epoch)
        else:
            avg_loss, sup_loss, con_loss, train_acc, train_auc = train_one_epoch_mean_teacher(
                model, train_loader, unlabeled_loader, optimizer, supervised_criterion, consistency_criterion, device, epoch
            )

        val_acc, val_auc, val_ap = validate(model, val_loader, device)
        scheduler.step()

        # Update history
        history["loss_total"].append(avg_loss)
        history["loss_sup"].append(sup_loss)
        history["loss_con"].append(con_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)
        history["val_ap"].append(val_ap)

        print(
            f"Epoch [{epoch + 1:02}/{epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc * 100:.2f}% "
            f"- Val Acc: {val_acc * 100:.2f}% - Val AUC: {val_auc:.4f} - Val AP: {val_ap:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = settings.EARLY_STOP_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

        # Cleanup
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return model, history
