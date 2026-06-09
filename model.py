import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset, Subset
from torchsummary import summary


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = nn.Sequential(  # excitation
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.excitation(self.squeeze(x)) * x  # recalibration


class SEPierceStageNet(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3):
        super().__init__()

        # Pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=None)

        # # Partially freeze backbone
        # for i, block in enumerate(self.backbone.features):
        #     if i < 6:  # freeze
        #         for param in block.parameters():
        #             param.requires_grad = False

        # Lightweight SE attention
        self.se = SEBlock(channels=1280, reduction=16)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.se(x)
        x = self.classifier(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SEPierceStageNet(num_classes=4, dropout=0.3).to(device)
summary(model, input_size=(3, 224, 224), device=device)
