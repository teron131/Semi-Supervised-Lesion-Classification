from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_history(history: dict[str, list[float]], output_dir: Path) -> Path:
    """Save history metrics to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss_total"]) + 1)
    csv_path = output_dir / "history.csv"
    keys = list(history)
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch," + ",".join(keys) + "\n")
        for idx, epoch in enumerate(epochs):
            row = [str(epoch)] + [f"{history[key][idx]:.6f}" for key in keys]
            handle.write(",".join(row) + "\n")
    return csv_path


def plot_history(history: dict[str, list[float]], output_dir: Path | None = None) -> Path | None:
    """Plot training history (loss, accuracy, AUC)."""
    epochs = range(1, len(history["loss_total"]) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["loss_total"], label="Total")
    plt.plot(epochs, history["loss_sup"], label="Supervised")
    unsup_key = "loss_unsup" if "loss_unsup" in history else "loss_con"
    plt.plot(epochs, history[unsup_key], label="Unsupervised")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["val_acc"], label="Val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    # Plot AUC
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_auc"], label="Train")
    plt.plot(epochs, history["val_auc"], label="Val")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    if output_dir is None:
        plt.show()
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "training_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path
