import gc
import itertools

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader


def train_one_epoch(
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
        consistency_loss = consistency_weight * consistency_criterion(unlabeled_preds_student, unlabeled_preds_teacher)

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


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Validate the student model."""
    model.student_model.eval()
    val_labels_list = []
    val_preds_list = []

    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        logits = model.student_model(imgs)
        preds = torch.sigmoid(logits)

        val_preds_list.extend(preds.cpu().numpy())
        val_labels_list.extend(targets.cpu().numpy())

    val_labels = np.array(val_labels_list)
    val_preds = np.array(val_preds_list)

    acc = accuracy_score(val_labels, np.round(val_preds))
    auc = roc_auc_score(val_labels, val_preds) if len(np.unique(val_labels)) > 1 else 0.0

    return acc, auc


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

    history = {"loss_total": [], "loss_sup": [], "loss_con": [], "train_acc": [], "train_auc": [], "val_acc": [], "val_auc": []}

    for epoch in range(epochs):
        avg_loss, sup_loss, con_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, unlabeled_loader, optimizer, supervised_criterion, consistency_criterion, device, epoch
        )

        val_acc, val_auc = validate(model, val_loader, device)
        scheduler.step()

        # Update history
        history["loss_total"].append(avg_loss)
        history["loss_sup"].append(sup_loss)
        history["loss_con"].append(con_loss)
        history["train_acc"].append(train_acc)
        history["train_auc"].append(train_auc)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"Epoch [{epoch + 1:02}/{epochs}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc * 100:.2f}% - Val Acc: {val_acc * 100:.2f}% - Val AUC: {val_auc:.4f}")

        # Cleanup
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    return model, history
