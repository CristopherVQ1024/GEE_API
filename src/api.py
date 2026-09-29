"""
api.py
======
Backend FastAPI que expone el modelo entrenado para inferencia sobre un
tile Sentinel-2 (GeoTIFF de 8 bandas, mismo formato que produce
src/gee_exporter.py). Es la pieza "FastAPI (backend/API que expone el
modelo)" de tu stack tecnológico.

Ejecutar (desde la raíz del repo):
    uvicorn src.api:app --reload --port 8000

Endpoints:
    GET  /health                -> chequeo simple
    POST /predict                -> sube un GeoTIFF de 8 bandas, devuelve
                                     la máscara de predicción como GeoTIFF
                                     (respuesta binaria) + estadísticas
                                     resumen en el header 'X-Prediction-Stats'
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from src.evaluate import ensemble_predict_proba, load_fold_checkpoints

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"

app = FastAPI(
    title="API de Detección de Tala Selectiva — Madre de Dios",
    description="Sirve el modelo U-Net entrenado para segmentar tala selectiva a partir de parches Sentinel-2.",
    version="0.1.0",
)

_state: dict = {"models": None, "cfg": None, "device": None}


def _load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.on_event("startup")
def load_models_on_startup() -> None:
    cfg = _load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        models = load_fold_checkpoints(cfg["paths"]["checkpoints"], cfg["model"], device)
    except FileNotFoundError:
        # Permite levantar la API sin checkpoints (útil para probar /health
        # mientras el entrenamiento aún no termina).
        models = None
    _state.update({"models": models, "cfg": cfg, "device": device})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": _state["models"] is not None,
        "n_models_in_ensemble": len(_state["models"]) if _state["models"] else 0,
        "device": str(_state["device"]),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Response:
    if _state["models"] is None:
        raise HTTPException(
            status_code=503,
            detail="No hay checkpoints entrenados en outputs/checkpoints. Corre src/train.py primero.",
        )

    contents = await file.read()
    with rasterio.MemoryFile(contents) as memfile:
        with memfile.open() as src:
            img = src.read().astype(np.float32)
            meta = src.meta.copy()

    if img.shape[0] != _state["cfg"]["model"]["in_channels"]:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {_state['cfg']['model']['in_channels']} bandas, "
                   f"se recibieron {img.shape[0]}.",
        )

    img = np.nan_to_num(img, nan=0.0)
    norm_stats = _state["models"][0]["norm_stats"]
    mean = np.array(norm_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.array(norm_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    img_norm = (img - mean) / np.clip(std, 1e-6, None)

    tensor = torch.from_numpy(img_norm).unsqueeze(0).float()
    probs = ensemble_predict_proba(_state["models"], tensor, _state["device"]).cpu().numpy()[0, 0]
    mask = (probs >= 0.5).astype(np.uint8)

    out_meta = meta.copy()
    out_meta.update(count=1, dtype="uint8", nodata=0)

    with rasterio.MemoryFile() as buffer:
        with buffer.open(**out_meta) as dst:
            dst.write(mask, 1)
        tiff_bytes = buffer.read()

    stats = {
        "area_afectada_pct": float(mask.mean() * 100),
        "n_pixeles_positivos": int(mask.sum()),
        "prob_promedio": float(probs.mean()),
    }

    return Response(
        content=tiff_bytes,
        media_type="image/tiff",
        headers={"X-Prediction-Stats": json.dumps(stats)},
    )
