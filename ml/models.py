"""Model A (baseline) and Model B (mask-supervised): one shared architecture class.

Both use the same grayscale ResNet-18 encoder (random init, no pretrained weights) and a
six-logit head. Model B adds a light U-Net style decoder that fuses the encoder's
stride-2/4/8/16/32 features and, in the last stage, the raw input image, so the mask
logits are produced at the full input resolution and keep the fine detail small defects need.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnet18

from .labels import NUM_CLASSES


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SixClassNet(nn.Module):
    def __init__(self, with_decoder: bool = False, num_classes: int = NUM_CLASSES, in_channels: int = 1):
        super().__init__()
        self.with_decoder = with_decoder
        r = resnet18(weights=None, num_classes=num_classes)  # never downloads weights
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.avgpool, self.fc = r.avgpool, r.fc
        if with_decoder:
            self.dec4 = _block(512 + 256, 128)  # stride 16
            self.dec3 = _block(128 + 128, 64)  # stride 8
            self.dec2 = _block(64 + 64, 48)  # stride 4
            self.dec1 = _block(48 + 64, 32)  # stride 2
            self.dec0 = _block(32 + in_channels, 16)  # stride 1 (fuses the input image)
            self.mask_head = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> dict:
        s0 = self.relu(self.bn1(self.conv1(x)))  # /2
        s1 = self.layer1(self.maxpool(s0))  # /4
        s2 = self.layer2(s1)  # /8
        s3 = self.layer3(s2)  # /16
        s4 = self.layer4(s3)  # /32
        logits = self.fc(torch.flatten(self.avgpool(s4), 1))
        out = {"logits": logits, "mask_logits": None}
        if self.with_decoder:
            d = self.dec4(torch.cat([self._up(s4, s3), s3], 1))
            d = self.dec3(torch.cat([self._up(d, s2), s2], 1))
            d = self.dec2(torch.cat([self._up(d, s1), s1], 1))
            d = self.dec1(torch.cat([self._up(d, s0), s0], 1))
            d = self.dec0(torch.cat([self._up(d, x), x], 1))
            out["mask_logits"] = self.mask_head(d)
        return out


def build_model(model_cfg: dict) -> SixClassNet:
    return SixClassNet(with_decoder=bool(model_cfg["with_decoder"]),
                       num_classes=int(model_cfg.get("num_classes", NUM_CLASSES)),
                       in_channels=int(model_cfg.get("in_channels", 1)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
