# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Qué es

Detección de tala selectiva en Madre de Dios (Perú) con Sentinel-2 + U-Net (ResNet34, `segmentation-models-pytorch`). Pipeline: extracción en Google Earth Engine → parches 256×256 → entrenamiento con K-fold espacial → evaluación → prototipo (FastAPI + Streamlit/Folium). Documentación en español. `README.md` está vacío; el texto real está en `README copy.md` (incluye el caveat sobre ground truth).

## Comandos

Siempre desde la raíz del repo y como módulos (`python -m src.x`), porque el código importa `from src....`. No hay tests ni linter configurados.

```bash
pip install -r requirements.txt

# 1. Folds espaciales (solo split_role == 'train'; el resto -> fold 'external_holdout')
python -m src.spatial_split --metadata <ruta>/patches_metadata.csv --out data/processed/patches_metadata.csv --block-km 10
#    sin --metadata corre una demo con datos sintéticos

# 2. Entrenar (CV espacial, un checkpoint por fold en outputs/checkpoints)
python -m src.train --metadata data/processed/patches_metadata.csv

# 3. Evaluar (ensamble de folds sobre el holdout externo)
python -m src.evaluate --metadata data/processed/patches_metadata.csv [--reference_raster ... --prediction_mosaic ...]

# Prototipo
uvicorn src.api:app --reload
streamlit run app_streamlit.py
```

## Arquitectura

- **Dataset v2 se genera fuera del repo**: en un notebook de Colab (`01_exploratory_analysis_v2`, no versionado aquí) con el proyecto EE de `config.yaml`. El repo solo consume el resultado: `data/processed/{images,masks}/<patch_id>.tif` + `patches_metadata.csv` + `bands.json`. 600 parches: 480 Tambopata (`train`, K-fold) y 120 Tahuamanu (`test_transfer`, solo prueba).
- **`src/gee_exporter.py` y `src/labels.py` son el flujo viejo de 8 bandas**; no producen el dataset v2. No los tomes como fuente de verdad del pipeline actual.
- **Acoplamiento de 15 canales**: `BAND_NAMES` en `src/dataset.py`, `model.in_channels` en `config/config.yaml` y el default de `src/models/unet.py` deben coincidir con `bands.json`. Cambiar uno exige cambiar los tres.
- **`config/config.yaml`** es la fuente única de hiperparámetros, rutas (`paths.*`), ROI/`study_areas` y K-fold; `train.py`, `evaluate.py` y `api.py` lo cargan cada uno con su propio `load_config()`.
- **Validación**: `spatial_split.py` asigna bloques geográficos (anti-leakage) a folds; `train.py` entrena un modelo por fold; `evaluate.py` carga todos los checkpoints (`load_fold_checkpoints`) y promedia probabilidades (`ensemble_predict_proba`). `api.py` reutiliza esas dos funciones de `evaluate.py` y arranca sin checkpoints (solo `/health` funciona).
- **Etiquetas débiles** (Hansen, JRC-TMF, RADD, MODIS): desbalance extremo (~0,5 % positivos) → pérdida BCE+Dice (`utils/losses.py`). El test real previsto es un subconjunto fotointerpretado a mano que nunca se usa para entrenar.

## Estado y trampas

- `data/` y `outputs/` están vacíos hasta que se copie el dataset desde Drive.
- Hay cambios locales sin commit en `dataset.py`, `spatial_split.py`, `train.py`.
- Pendiente de la investigación: fotointerpretar 50–100 parches, verificar endmembers NDFI, reemplazar NaN por 0 tras normalizar.
