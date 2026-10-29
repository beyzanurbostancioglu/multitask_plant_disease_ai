import timm
import torch
import torch.nn as nn

from class_map import PLANTS, DISEASE_KEYS


class MultiTaskNet(nn.Module):
    """Ortak omurga + 3 başlık: bitki türü, hastalık, şiddet (sağlıklı/genel/ciddi)."""

    def __init__(self, backbone="efficientnet_b0", pretrained=True, drop=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        f = self.backbone.num_features
        mk = lambda n: nn.Sequential(nn.Dropout(drop), nn.Linear(f, n))
        self.head_plant = mk(len(PLANTS))
        self.head_disease = mk(len(DISEASE_KEYS))
        self.head_sev = mk(3)

    def forward(self, x):
        f = self.backbone(x)               # GAP uygulanmış öznitelik vektörü
        return self.head_plant(f), self.head_disease(f), self.head_sev(f)
