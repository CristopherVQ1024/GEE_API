"""
utils/losses.py
================
Pérdida compuesta BCEWithLogits + Dice, necesaria porque la clase "tala
selectiva" suele ocupar menos del 5% del área total de un parche: con solo
BCE el modelo puede converger a predecir "todo bosque sano" y aun así
obtener una pérdida baja. Dice Loss penaliza directamente el mal solape de
la clase minoritaria y compensa ese desbalance.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.reshape(probs.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    """
    L_total = bce_weight * BCEWithLogitsLoss + dice_weight * DiceLoss

    pos_weight: opcional, escalar (o tensor de 1 elemento) para ponderar
    aún más la clase positiva dentro de BCE si el desbalance es extremo
    (p.ej. pos_weight = (n_negativos / n_positivos) calculado sobre tu
    set de entrenamiento).
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
