"""
utils/metrics.py
=================
Métricas de evaluación a nivel de píxel, calculadas ESTRICTAMENTE sobre la
clase positiva ("tala selectiva"), tal como exige tu metodología: IoU
(Jaccard), Precision, Recall, F1-score. Funcionan tanto para evaluar contra
tu propio test set (spatial k-fold) como para la validación externa
(comparar predicción binaria vs. polígonos oficiales rasterizados).
"""
from __future__ import annotations

import numpy as np
import torch


def _to_binary(preds: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Convierte logits o probabilidades a máscara binaria."""
    if preds.dtype != torch.bool:
        # Si vienen como logits (pueden ser negativos), pasamos por sigmoid.
        if (preds.min() < 0) or (preds.max() > 1):
            preds = torch.sigmoid(preds)
        preds = (preds >= threshold).float()
    return preds


def confusion_counts(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> dict:
    """
    Cuenta TP/FP/FN/TN sumados sobre el batch completo (todos los píxeles).
    Sumar conteos crudos antes de calcular ratios (en vez de promediar
    métricas por imagen) es más robusto cuando hay imágenes sin ningún
    píxel positivo.
    """
    preds_bin = _to_binary(preds, threshold)
    targets_bin = (targets >= 0.5).float()

    tp = (preds_bin * targets_bin).sum().item()
    fp = (preds_bin * (1 - targets_bin)).sum().item()
    fn = ((1 - preds_bin) * targets_bin).sum().item()
    tn = ((1 - preds_bin) * (1 - targets_bin)).sum().item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict, eps: float = 1e-7) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


class RunningConfusionCounts:
    """Acumula TP/FP/FN/TN a través de todos los batches de una época/evaluación."""

    def __init__(self):
        self.totals = {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}

    def update(self, preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> None:
        batch_counts = confusion_counts(preds, targets, threshold)
        for k in self.totals:
            self.totals[k] += batch_counts[k]

    def compute(self) -> dict:
        return metrics_from_counts(self.totals)


def compare_binary_rasters(pred_mask: np.ndarray, reference_mask: np.ndarray) -> dict:
    """
    Versión NumPy pura (sin torch) para comparar un raster de predicción
    contra un raster de referencia externo ya rasterizado al mismo grid
    (p.ej. polígonos de GeoBosques/GFW vía src/labels.rasterize_reference_polygons).
    Útil en evaluate.py para la validación empírica externa.
    """
    pred = (pred_mask > 0.5).astype(np.uint8)
    ref = (reference_mask > 0.5).astype(np.uint8)

    tp = int(np.sum((pred == 1) & (ref == 1)))
    fp = int(np.sum((pred == 1) & (ref == 0)))
    fn = int(np.sum((pred == 0) & (ref == 1)))
    tn = int(np.sum((pred == 0) & (ref == 0)))

    return metrics_from_counts({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
