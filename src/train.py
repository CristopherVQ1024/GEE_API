"""
train.py
========
Script principal de entrenamiento con Spatial K-Fold cross-validation.

Flujo esperado:
  1. Ya exportaste tus parches (imágenes 8-banda + máscaras) desde GEE y
     los descargaste a data/processed/images y data/processed/masks
     (ver src/gee_exporter.py y src/labels.py).
  2. Generaste un CSV de metadatos con columnas:
       patch_id, lon_center, lat_center
     (uno por cada parche exportado — el centro de cada tile lo puedes
     sacar de la lista `tiles` que devuelve build_export_grid()).
  3. Corriste src/spatial_split.py sobre ese CSV para añadir `fold` y
     guardaste el resultado, p.ej. en data/processed/patches_metadata.csv

Uso:
    python -m src.train --metadata data/processed/patches_metadata.csv

El fold 'external_holdout' NUNCA se usa aquí — se reserva para
evaluate.py (validación empírica final, sección "Resultados" del artículo).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import SelectiveLoggingDataset, build_paths_from_ids, compute_normalization_stats
from src.models.unet import build_unet
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import RunningConfusionCounts

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_fold(cfg: dict, metadata: pd.DataFrame, val_fold: int, device: torch.device) -> dict:
    """
    Entrena con todos los folds numéricos != val_fold (excluyendo siempre
    'external_holdout') y valida en val_fold. Devuelve las métricas de
    validación y la ruta del mejor checkpoint para este fold.
    """
    cv_pool = metadata[metadata["fold"] != "external_holdout"].copy()
    train_df = cv_pool[cv_pool["fold"] != val_fold]
    val_df = cv_pool[cv_pool["fold"] == val_fold]

    images_dir = cfg["paths"]["patches_images"]
    masks_dir = cfg["paths"]["patches_masks"]

    train_img_paths, train_mask_paths = build_paths_from_ids(train_df["patch_id"].tolist(), images_dir, masks_dir)
    val_img_paths, val_mask_paths = build_paths_from_ids(val_df["patch_id"].tolist(), images_dir, masks_dir)

    norm_stats = compute_normalization_stats(train_img_paths)

    train_ds = SelectiveLoggingDataset(train_img_paths, train_mask_paths, normalize_stats=norm_stats)
    val_ds = SelectiveLoggingDataset(val_img_paths, val_mask_paths, normalize_stats=norm_stats)

    train_cfg = cfg["model"]["training"]
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True,
                               num_workers=train_cfg["num_workers"], drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False,
                             num_workers=train_cfg["num_workers"])

    model_cfg = cfg["model"]
    model = build_unet(
        encoder_name=model_cfg["encoder"],
        encoder_weights=model_cfg["encoder_weights"],
        in_channels=model_cfg["in_channels"],
        classes=model_cfg["classes"],
    ).to(device)

    loss_fn = BCEDiceLoss(bce_weight=model_cfg["loss"]["bce_weight"], dice_weight=model_cfg["loss"]["dice_weight"])

    opt_cfg = model_cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

    sched_cfg = model_cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=sched_cfg["factor"], patience=sched_cfg["patience"]
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"unet_fold{val_fold}_best.pt"

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / len(train_ds)

        model.eval()
        val_loss_total = 0.0
        val_counts = RunningConfusionCounts()
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                val_loss_total += loss.item() * imgs.size(0)
                val_counts.update(torch.sigmoid(logits), masks)
        val_loss = val_loss_total / max(len(val_ds), 1)
        val_metrics = val_counts.compute()

        scheduler.step(val_loss)

        print(
            f"[fold {val_fold}] epoch {epoch + 1}/{train_cfg['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"IoU={val_metrics['iou']:.4f} F1={val_metrics['f1']:.4f} "
            f"Prec={val_metrics['precision']:.4f} Rec={val_metrics['recall']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, best_ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"[fold {val_fold}] Early stopping en epoch {epoch + 1}")
                break

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    return {"fold": val_fold, "checkpoint": str(best_ckpt_path), **best_ckpt["val_metrics"]}


def run_spatial_cross_validation(cfg: dict, metadata_path: str) -> None:
    metadata = pd.read_csv(metadata_path)
    device = get_device()
    print(f"Usando device: {device}")

    numeric_folds = sorted(
        [f for f in metadata["fold"].unique() if f != "external_holdout"],
        key=lambda x: int(x),
    )

    results = []
    for fold in numeric_folds:
        result = train_one_fold(cfg, metadata, val_fold=int(fold), device=device)
        results.append(result)

    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.DataFrame(results)
    df_results.to_csv(metrics_dir / "cv_results_per_fold.csv", index=False)

    summary = {
        metric: {"mean": float(np.mean(df_results[metric])), "std": float(np.std(df_results[metric]))}
        for metric in ["iou", "f1", "precision", "recall"]
    }
    with open(metrics_dir / "cv_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Resumen validación cruzada espacial (n_folds) ===")
    for metric, stats in summary.items():
        print(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\nResultados guardados en {metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True,
                         help="CSV con patch_id, lon_center, lat_center, fold (ver src/spatial_split.py)")
    args = parser.parse_args()

    cfg = load_config()
    run_spatial_cross_validation(cfg, args.metadata)
