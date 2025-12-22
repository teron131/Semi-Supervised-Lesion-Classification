import copy
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet50_Weights, convnext_tiny as _convnext_tiny, resnet50 as _resnet50


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock | Bottleneck], layers: list[int], use_fc: bool = False, dropout: float | None = None):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.use_fc = use_fc
        if self.use_fc:
            self.fc_add = nn.Linear(512 * block.expansion, 512)

        self.dropout = nn.Dropout(p=dropout) if dropout else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block: type[BasicBlock | Bottleneck], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.mean(dim=(-2, -1))
        x = x.view(x.size(0), -1)
        if self.use_fc:
            x = F.relu(self.fc_add(x))
        if self.dropout:
            x = self.dropout(x)
        return x


class ClassifierModel(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, num_classes: int = 1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.classifier(x)
        return x


def get_resnet50(pre_trained: bool = True, dropout: float | None = 0.5) -> ResNet:
    model = ResNet(Bottleneck, [3, 4, 6, 3], dropout=dropout)
    if pre_trained:
        weights = ResNet50_Weights.IMAGENET1K_V2
        pre_trained_model = _resnet50(weights=weights)
        state_dict = pre_trained_model.state_dict()
        new_weights = {k: state_dict[k] for k in model.state_dict() if k in state_dict}
        model.load_state_dict(new_weights, strict=False)
    return model


def get_convnext_tiny(pre_trained: bool = True) -> tuple[nn.Module, int]:
    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pre_trained else None
    model = _convnext_tiny(weights=weights)
    feature_dim = model.classifier[2].in_features
    model.classifier = nn.Identity()
    backbone = nn.Sequential(model, nn.Flatten(1))
    return backbone, feature_dim


class MeanTeacherModel(nn.Module):
    def __init__(self, student_model: nn.Module, ema_decay: float, consistency_max_weight: float, rampup_epochs: int):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = copy.deepcopy(student_model)
        self.ema_decay = ema_decay
        self.consistency_max_weight = consistency_max_weight
        self.rampup_epochs = rampup_epochs

        # Teacher model parameters are not trained via backprop
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.student_model(x)

    def update_teacher_model(self, current_epoch: int):
        """Update teacher model parameters using Exponential Moving Average."""
        momentum = min(1 - 1 / (current_epoch + 1), self.ema_decay)
        with torch.no_grad():
            for s_param, t_param in zip(self.student_model.parameters(), self.teacher_model.parameters(), strict=False):
                t_param.data.mul_(momentum).add_((1 - momentum) * s_param.data)

    def sigmoid_rampup(self, current_epoch: int) -> float:
        """Consistency weight ramp-up function."""
        if self.rampup_epochs <= 0:
            return 1.0
        epoch = np.clip(current_epoch, 0.0, float(self.rampup_epochs))
        phase = 1.0 - epoch / float(self.rampup_epochs)
        return float(np.exp(-5.0 * phase * phase))

    def get_consistency_weight(self, current_epoch: int) -> float:
        """Get the current consistency loss weight."""
        return self.consistency_max_weight * self.sigmoid_rampup(current_epoch)
