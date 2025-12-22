import torch
from torch import nn
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    """Focal loss built on logits for numerical stability.

    Alpha should be about the ratio of the classes, ratio = alpha : 1 - alpha
    e.g. alpha = 0.75 for 3:1 ratio
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.6, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
