"""
evaluate.py
===========
Validación final del proyecto, en dos niveles (ver README):

  1. Validación en el bloque geográfico 'external_holdout' generado por
     src/spatial_split.py: nunca visto en entrenamiento ni en el ajuste de
     hiperparámetros de ningún fold. Se evalúa con un ENSEMBLE (promedio de
     probabilidades) de los N modelos entrenados por fold en train.py —
     más robusto que un solo modelo y además es tu modelo final para el
     prototipo/API.

  2. Validación empírica externa: compara el mosaico de predicciones
     contra un raster de referencia oficial ya alineado al mismo grid
     (GeoBosques si obtienes acceso, GFW, o tu subconjunto validado por
     fotointerpretación — ver src/labels.py). Esta es la comparación que
     sustenta tu aporte diferenciador de "validación empírica aplicada".

Uso:
    python -m src.evaluate --metadata data/processed/patches_metadata.csv
    python -m src.evaluate --metadata ... --reference_raster path/al/raster_oficial_alineado.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml

from src.dataset import SelectiveLoggingDataset, build_paths_from_ids
from src.models.unet import build_unet
from src.utils.metrics import RunningConfusionCounts, compare_binary_rasters

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_fold_checkpoints(checkpoints_dir: str, model_cfg: dict, device: torch.device) -> list[dict]:
    """Carga todos los checkpoints unet_fold*_best.pt guardados por train.py."""
    ckpt_paths = sorted(Path(checkpoints_dir).glob("unet_fold*_best.pt"))
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en {checkpoints_dir}. Corre src/train.py primero."
        )

    loaded = []
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location=device)
        model = build_unet(
            encoder_name=model_cfg["encoder"],
            encoder_weights=None,  # los pesos vienen del checkpoint, no de ImageNet de nuevo
            in_channels=model_cfg["in_channels"],
            classes=model_cfg["classes"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        loaded.append({"model": model, "norm_stats": ckpt["norm_stats"], "path": str(p)})
    return loaded


def ensemble_predict_proba(models: list[dict], img: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Promedia las probabilidades (no los logits) de todos los modelos del ensemble."""
    probs_sum = None
    for m in models:
        with torch.no_grad():
            logits = m["model"](img.to(device))
            probs = torch.sigmoid(logits)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return probs_sum / len(models)


def evaluate_external_holdout(cfg: dict, metadata: pd.DataFrame, device: torch.device) -> dict:
    holdout_df = metadata[metadata["fold"] == "external_holdout"].copy()
    if holdout_df.empty:
        raise ValueError("No hay parches con fold == 'external_holdout' en el CSV de metadatos.")

    images_dir = cfg["paths"]["patches_images"]
    masks_dir = cfg["paths"]["patches_masks"]
    img_paths, mask_paths = build_paths_from_ids(holdout_df["patch_id"].tolist(), images_dir, masks_dir)

    models = load_fold_checkpoints(cfg["paths"]["checkpoints"], cfg["model"], device)
    # Usamos las normalize_stats del primer fold como aproximación razonable
    # (en rigor podrías promediar las stats de todos los folds de train).
    norm_stats = models[0]["norm_stats"]

    ds = SelectiveLoggingDataset(img_paths, mask_paths, normalize_stats=norm_stats)

    counts = RunningConfusionCounts()
    for img, mask in ds:
        img_batch = img.unsqueeze(0)
        probs = ensemble_predict_proba(models, img_batch, device).cpu()
        counts.update(probs, mask.unsqueeze(0))

    return counts.compute()


def predict_and_export_mask(cfg: dict, models: list[dict], image_path: str, out_path: str,
                             device: torch.device, threshold: float = 0.5) -> str:
    """
    Corre el ensemble sobre un único tile GeoTIFF y guarda la máscara de
    predicción binaria como GeoTIFF (mismo grid/CRS que la imagen de
    entrada) — esto es lo que consume el backend FastAPI/Streamlit del
    prototipo para pintar el mapa interactivo.
    """
    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)
        img = np.nan_to_num(img, nan=0.0)
        meta = src.meta.copy()

    norm_stats = models[0]["norm_stats"]
    mean = np.array(norm_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(norm_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    img_norm = (img - mean) / np.clip(std, 1e-6, None)

    tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
    probs = ensemble_predict_proba(models, tensor, device).cpu().numpy()[0, 0]
    mask = (probs >= threshold).astype(np.uint8)

    meta.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask, 1)

    return out_path


def evaluate_against_external_reference(prediction_raster_path: str, reference_raster_path: str) -> dict:
    """
    Compara un mosaico de predicción ya exportado (ver predict_and_export_mask,
    o un mosaico construido uniendo varios tiles) contra un raster de
    referencia oficial previamente alineado al MISMO grid/resolución/CRS
    (usa src/labels.rasterize_reference_polygons para prepararlo).
    """
    with rasterio.open(prediction_raster_path) as src:
        pred = src.read(1)
    with rasterio.open(reference_raster_path) as src:
        ref = src.read(1)

    if pred.shape != ref.shape:
        raise ValueError(
            f"Las formas no coinciden: predicción {pred.shape} vs referencia {ref.shape}. "
            "Asegúrate de rasterizar la referencia al mismo grid con labels.rasterize_reference_polygons."
        )

    return compare_binary_rasters(pred, ref)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--reference_raster", type=str, default=None,
                         help="Opcional: raster oficial ya alineado, para validación empírica externa")
    parser.add_argument("--prediction_mosaic", type=str, default=None,
                         help="Requerido si se pasa --reference_raster")
    args = parser.parse_args()

    cfg = load_config()
    device = get_device()
    metadata = pd.read_csv(args.metadata)

    print("=== 1. Validación en bloque geográfico reservado (external_holdout) ===")
    holdout_metrics = evaluate_external_holdout(cfg, metadata, device)
    for k, v in holdout_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    metrics_dir = Path(cfg["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "external_holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(holdout_metrics, f, indent=2)

    if args.reference_raster:
        if not args.prediction_mosaic:
            raise SystemExit("Debes pasar --prediction_mosaic junto con --reference_raster")
        print("\n=== 2. Validación empírica externa (vs. dataset oficial) ===")
        ext_metrics = evaluate_against_external_reference(args.prediction_mosaic, args.reference_raster)
        for k, v in ext_metrics.items():
            print(f"{k.upper()}: {v:.4f}")
        with open(metrics_dir / "external_reference_metrics.json", "w", encoding="utf-8") as f:
            json.dump(ext_metrics, f, indent=2)
