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


def plot_history(history: dict[str, list[float]]):
    """Plot training history (loss, accuracy, AUC)."""
    epochs = range(1, len(history["loss_total"]) + 1)

    # Plot Loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["loss_total"], label="Total")
    plt.plot(epochs, history["loss_sup"], label="Supervised")
    plt.plot(epochs, history["loss_con"], label="Consistency")
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
    plt.show()
