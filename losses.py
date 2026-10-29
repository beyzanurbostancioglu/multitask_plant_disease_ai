import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (+ opsiyonel sınıf ağırlığı). ignore_index=-1: şiddet etiketi olmayan örnekler."""

    def __init__(self, gamma=2.0, weight=None, ignore_index=-1, label_smoothing=0.0):
        super().__init__()
        self.gamma, self.ii, self.ls = gamma, ignore_index, label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, y):
        mask = y != self.ii
        if mask.sum() == 0:
            return logits.sum() * 0.0
        logits, y = logits[mask], y[mask]
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction="none",
                             label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return (((1 - pt) ** self.gamma) * ce).mean()


def class_weights(labels, n_classes, power=0.5):
    """Ters frekansın kökü (aşırı agresif olmasın diye), ortalama=1 normalize."""
    cnt = torch.bincount(labels[labels >= 0], minlength=n_classes).float().clamp(min=1)
    w = (1.0 / cnt) ** power
    return w / w.mean()
