import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Định nghĩa ConvNormReLUBlock
class ConvNormReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, activation=nn.ReLU6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation() if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

# Định nghĩa Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.depthwise = ConvNormReLUBlock(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels)
        self.pointwise = ConvNormReLUBlock(in_channels, out_channels, kernel_size=1, activation=nn.Identity)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Định nghĩa Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.residual = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvNormReLUBlock(in_channels, hidden_dim, kernel_size=1))
        layers.append(DepthwiseSeparableConv(hidden_dim, out_channels, stride))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        return self.conv(x)

# Định nghĩa MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2):
        super().__init__()
        # Config từ MobileNetV2 paper (in_channels, expand_ratio, out_channels, repeat, stride)
        self.cfgs = [
            (32, 1, 16, 1, 1),
            (16, 6, 24, 2, 2),
            (24, 6, 32, 3, 2),
            (32, 6, 64, 4, 2),
            (64, 6, 96, 3, 1),
            (96, 6, 160, 3, 2),
            (160, 6, 320, 1, 1),
        ]

        # First layer
        self.first_conv = ConvNormReLUBlock(3, 32, kernel_size=3, stride=2, padding=1)

        # Feature extraction layers
        layers = []
        in_channels = 32
        for in_ch, t, c, n, s in self.cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(in_channels, c, stride, t))
                in_channels = c
        self.features = nn.Sequential(*layers)

        # Last layers
        self.last_conv = ConvNormReLUBlock(in_channels, 1280, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.features(x)
        x = self.last_conv(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# Tạo mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV2(num_classes=10).to(device)