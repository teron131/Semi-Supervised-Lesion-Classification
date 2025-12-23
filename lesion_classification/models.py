import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny as _convnext_tiny


class ClassifierModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        feature_dim: int,
        num_classes: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def get_convnext_tiny(pre_trained: bool = True) -> tuple[nn.Module, int]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pre_trained else None
    model = _convnext_tiny(weights=weights)
    feature_dim = model.classifier[2].in_features
    model.classifier = nn.Identity()
    backbone = nn.Sequential(model, nn.Flatten(1))
    return backbone, feature_dim
