import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny as _convnext_tiny, resnet50 as _resnet50


class ClassifierModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def get_convnext_tiny(pre_trained: bool = True) -> tuple[nn.Module, int]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pre_trained else None
    model = _convnext_tiny(weights=weights)
    feature_dim = model.classifier[2].in_features
    model.classifier = nn.Identity()
    backbone = nn.Sequential(model, nn.Flatten(1))
    return backbone, feature_dim


def get_resnet50(pre_trained: bool = True) -> tuple[nn.Module, int]:
    weights = ResNet50_Weights.IMAGENET1K_V2 if pre_trained else None
    model = _resnet50(weights=weights)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    backbone = nn.Sequential(model, nn.Flatten(1))
    return backbone, feature_dim
