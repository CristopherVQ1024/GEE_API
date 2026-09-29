"""
dataset.py
==========
Dataset de PyTorch para parches de Sentinel-2 (8 bandas) + máscara binaria
de tala selectiva/degradación (ver src/labels.py para cómo se genera esa
máscara).

Espera que, tras exportar desde GEE (src/gee_exporter.py) y descargar de
Drive, tengas dos carpetas paralelas con el mismo nombre de archivo por
parche:
    data/processed/images/tile_r000_c012.tif   (8 bandas, float32)
    data/processed/masks/tile_r000_c012.tif    (1 banda, uint8, 0/1)

Y opcionalmente un CSV de metadatos (uno por parche) con al menos:
    patch_id, lon_center, lat_center, fold
generado por src/spatial_split.py, para poder filtrar por fold en
train.py/evaluate.py sin tener que volver a tocar los rasters.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# Debe coincidir con el orden de bandas producido en gee_exporter.build_feature_stack
BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "sigma_NIR"]


def _read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


class SelectiveLoggingDataset(Dataset):
    """
    image_paths, mask_paths: listas alineadas (misma longitud, mismo orden)
        con las rutas a cada parche de imagen/máscara.
    transform: callable estilo albumentations que recibe/retorna
        {'image': HxWxC, 'mask': HxWx1} — ver notebooks para un ejemplo de
        pipeline de aumentación (flip/rotación; EVITA aumentos que alteren
        el valor espectral como brightness/contrast agresivo, ya que aquí
        el valor físico de reflectancia importa).
    normalize_stats: dict opcional {"mean": [C], "std": [C]} calculado
        sobre el set de entrenamiento (ver utils/compute_norm_stats en el
        notebook exploratorio) para normalizar cada banda.
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_paths: list[str],
        transform: Optional[Callable] = None,
        normalize_stats: Optional[dict] = None,
    ):
        assert len(image_paths) == len(mask_paths), "image_paths y mask_paths deben tener la misma longitud"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.normalize_stats = normalize_stats

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = _read_raster(self.image_paths[idx])          # (C, H, W)
        mask = _read_raster(self.mask_paths[idx])           # (1, H, W) o (H, W)

        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # Reemplaza NaN (bordes sin datos por nubes/mosaico) por 0 tras
        # normalizar, para no romper el entrenamiento.
        img = np.nan_to_num(img, nan=0.0)

        if self.normalize_stats is not None:
            mean = np.array(self.normalize_stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
            std = np.array(self.normalize_stats["std"], dtype=np.float32).reshape(-1, 1, 1)
            img = (img - mean) / np.clip(std, 1e-6, None)

        if self.transform is not None:
            # albumentations espera HWC
            img_hwc = np.transpose(img, (1, 2, 0))
            mask_hwc = np.transpose(mask, (1, 2, 0))
            augmented = self.transform(image=img_hwc, mask=mask_hwc)
            img = np.transpose(augmented["image"], (2, 0, 1))
            mask = np.transpose(augmented["mask"], (2, 0, 1))

        return torch.from_numpy(img.copy()).float(), torch.from_numpy(mask.copy()).float()


def build_paths_from_ids(patch_ids: list[str], images_dir: str, masks_dir: str,
                          ext: str = ".tif") -> tuple[list[str], list[str]]:
    """Convierte una lista de patch_id en rutas de imagen/máscara, asumiendo el layout estándar."""
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    image_paths = [str(images_dir / f"{pid}{ext}") for pid in patch_ids]
    mask_paths = [str(masks_dir / f"{pid}{ext}") for pid in patch_ids]
    return image_paths, mask_paths


def compute_normalization_stats(image_paths: list[str]) -> dict:
    """
    Calcula media/desviación estándar por banda sobre un conjunto de
    parches (usa SOLO parches de entrenamiento, nunca de validación/test,
    para no filtrar información). Devuelve {"mean": [...], "std": [...]}.
    """
    sums, sq_sums, count = None, None, 0
    for p in image_paths:
        img = _read_raster(p)  # (C, H, W)
        c = img.shape[0]
        if sums is None:
            sums = np.zeros(c, dtype=np.float64)
            sq_sums = np.zeros(c, dtype=np.float64)
        flat = img.reshape(c, -1)
        sums += np.nansum(flat, axis=1)
        sq_sums += np.nansum(flat ** 2, axis=1)
        count += flat.shape[1]

    mean = sums / count
    var = sq_sums / count - mean ** 2
    std = np.sqrt(np.clip(var, 1e-12, None))
    return {"mean": mean.tolist(), "std": std.tolist()}
