import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Định nghĩa Squeeze-and-Excitation Block
class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        reduced_channels = max(in_channels // 4, 1)  # Giảm channels như trong paper
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, reduced_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        identity = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.hardsigmoid(x)
        return identity * x.view(x.size(0), -1, 1, 1)

# Định nghĩa Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_se=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.hardswish = nn.Hardswish(inplace=True)
        self.se = SqueezeExcitation(in_channels) if use_se else nn.Identity()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.hardswish(out)
        out = self.se(out)
        out = self.pointwise(out)
        out = self.bn2(out)
        return out

# Định nghĩa Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_se=False, activation=nn.Hardswish):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(activation(inplace=True))
        layers.append(DepthwiseSeparableConv(hidden_dim, out_channels, stride, use_se=use_se))
        self.conv = nn.Sequential(*layers)
        self.activation = activation(inplace=True) if not use_se else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        if self.use_res_connect:
            return x + out
        return out

# Định nghĩa MobileNetV3
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10, config="large"):
        super().__init__()
        # Config từ MobileNetV3 paper (in_channels, expand_ratio, out_channels, num_blocks, stride, use_se, activation)
        configs_dict = {
            "large": [
                (16, 1, 16, 1, 1, False, nn.ReLU),
                (16, 4, 24, 2, 2, False, nn.ReLU),
                (24, 3, 24, 1, 1, False, nn.ReLU),
                (24, 3, 40, 3, 2, True, nn.ReLU),
                (40, 3, 40, 2, 1, True, nn.ReLU),
                (40, 6, 80, 4, 2, False, nn.Hardswish),
                (80, 2.5, 80, 3, 1, False, nn.Hardswish),
                (80, 6, 112, 2, 1, True, nn.Hardswish),
                (112, 6, 160, 3, 2, True, nn.Hardswish),
            ],
            "small": [
                (16, 1, 16, 1, 2, True, nn.ReLU),
                (16, 4.5, 24, 2, 2, False, nn.ReLU),
                (24, 3.67, 40, 3, 2, True, nn.Hardswish),
                (40, 4, 48, 2, 1, True, nn.Hardswish),
                (48, 6, 96, 3, 2, True, nn.Hardswish),
            ]
        }
        self.cfgs = configs_dict[config]

        # First layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )

        # Feature extraction layers
        layers = []
        in_channels = 16
        for in_ch, t, c, n, s, se, act in self.cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, c, stride, t, use_se=se, activation=act))
                in_channels = c
        self.features = nn.Sequential(*layers)

        # Last layers
        last_channels = 160 if config == "large" else 96
        hidden_channels = 960 if config == "large" else 576
        out_channels = 1280 if config == "large" else 1024
        self.last_conv = nn.Sequential(
            nn.Conv2d(last_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3(num_classes=10, config="large").to(device)