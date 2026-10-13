"""Comparación visual de la predicción del modelo en uno o varios parches.

Paneles: RGB | falso color SWIR | dNDFI | etiqueta v3 | probabilidad | predicción (> umbral) sobre RGB.
Tambopata se predice con el modelo del fold fuera de muestra; Tahuamanu con el ensamble.

    python -m src.visualize_patch tambopata_r015_c017 tahuamanu_r000_c001
    python -m src.visualize_patch --positivos          # todos los parches fotointerpretados 'si'
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from src.dataset import BAND_NAMES
from src.evaluate import get_device, load_config, load_fold_checkpoints
from src.evaluate_patch_level import final_labels, predict


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def composite(img, names):
    return np.dstack([stretch(np.nan_to_num(img[BAND_NAMES.index(n)])) for n in names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patch_ids", nargs="*")
    ap.add_argument("--positivos", action="store_true", help="usar los parches etiquetados 'si'")
    ap.add_argument("--ckpt", default=None, help="por defecto paths.checkpoints de config.yaml")
    ap.add_argument("--metadata", default="data/processed/patches_metadata.csv")
    ap.add_argument("--masks_dir", default="data/processed/masks_v3")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/predictions")
    a = ap.parse_args()

    ids = list(a.patch_ids)
    if a.positivos:
        lab = final_labels()
        ids += lab[lab.label == "si"].patch_id.tolist()
    if not ids:
        ap.error("indica patch_ids o --positivos")

    cfg, dev = load_config(), get_device()
    models = load_fold_checkpoints(a.ckpt or cfg["paths"]["checkpoints"], cfg["model"], dev)
    meta = pd.read_csv(a.metadata).set_index("patch_id")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pid in ids:
        path = f"{cfg['paths']['patches_images']}/{pid}.tif"
        with rasterio.open(path) as s:
            img = s.read().astype("float32")
        fold = meta.loc[pid, "fold"]
        if fold == "external_holdout":
            p = np.mean([predict(m, path, dev) for m in models], axis=0)
            quien = "ensamble de 5 folds"
        else:
            p = predict(models[int(fold)], path, dev)
            quien = f"modelo fold {int(fold)} (fuera de muestra)"
        mask_path = Path(a.masks_dir) / f"{pid}.tif"
        m = rasterio.open(mask_path).read(1) if mask_path.exists() else np.zeros(p.shape)

        rgb = composite(img, ["B4", "B3", "B2"])
        pred = p > a.threshold
        over = rgb.copy()
        over[pred] = 0.5 * over[pred] + 0.5 * np.array([1.0, 0.1, 0.1])

        fig, ax = plt.subplots(1, 6, figsize=(24, 4.4))
        ax[0].imshow(rgb); ax[0].set_title("RGB (t1)")
        ax[1].imshow(composite(img, ["B12", "B8", "B4"])); ax[1].set_title("Falso color SWIR")
        d = ax[2].imshow(img[BAND_NAMES.index("dNDFI")], cmap="RdBu", vmin=-0.5, vmax=0.5)
        ax[2].set_title("dNDFI (rojo = caída)"); fig.colorbar(d, ax=ax[2], fraction=0.046)
        ax[3].imshow(rgb); ax[3].imshow(np.ma.masked_where(m < 1, m), cmap="autumn", alpha=0.9)
        ax[3].set_title(f"Etiqueta v3 ({int((m > 0).sum())} px)")
        h = ax[4].imshow(p, cmap="magma", vmin=0, vmax=1)
        ax[4].set_title(f"Probabilidad (máx {p.max():.2f})"); fig.colorbar(h, ax=ax[4], fraction=0.046)
        ax[5].imshow(over); ax[5].set_title(f"Predicción > {a.threshold} ({int(pred.sum())} px)")
        for x in ax:
            x.axis("off")
        fig.suptitle(f"{pid} — {quien}", y=1.02)
        fig.tight_layout()
        dst = out_dir / f"vis_{pid}.png"
        fig.savefig(dst, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(dst)


if __name__ == "__main__":
    main()
