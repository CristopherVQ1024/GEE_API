"""
spatial_split.py
=================
Validación cruzada espacial (Spatial K-Fold) por bloques geográficos.

Por qué es necesario: si divides los parches de 256x256 al azar, dos
parches vecinos casi siempre terminan uno en train y otro en test. Como
son espacialmente autocorrelacionados (comparten condiciones de iluminación,
suelo, fenología, ruido de sensor), el modelo puede reportar métricas
artificialmente altas sin haber aprendido nada generalizable ("spatial
leakage"). La solución estándar en teledetección/ecología es agrupar los
parches en BLOQUES geográficos grandes (aquí, 20x20 km) y asignar bloques
completos —nunca parches individuales— a cada fold.

Este módulo es puro Python/NumPy/Pandas: no depende de GEE, así que corre
igual en tu entorno de entrenamiento (Colab/Kaggle/local) usando solo los
metadatos de los parches ya exportados (su centroide lon/lat).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpatialKFoldConfig:
    n_folds: int = 5
    block_size_km: int = 20
    random_seed: int = 42
    held_out_external_fold: bool = True


def _km_to_deg(block_size_km: float, lat_ref_deg: float) -> tuple[float, float]:
    """Convierte un tamaño de bloque en km a grados (lon, lat) en lat_ref_deg."""
    deg_lat = block_size_km / 111.32
    deg_lon = block_size_km / (111.32 * math.cos(math.radians(lat_ref_deg)))
    return deg_lon, deg_lat


def assign_block_ids(
    patches_df: pd.DataFrame,
    lon_col: str = "lon_center",
    lat_col: str = "lat_center",
    block_size_km: int = 20,
) -> pd.Series:
    """
    Asigna a cada parche un `block_id` (string "r{row}_c{col}") según en
    qué celda de una grilla de block_size_km x block_size_km cae su
    centroide. Todos los parches con el mismo block_id SIEMPRE van al
    mismo fold.
    """
    lat_ref = patches_df[lat_col].mean()
    deg_lon, deg_lat = _km_to_deg(block_size_km, lat_ref)

    lon_min = patches_df[lon_col].min()
    lat_min = patches_df[lat_col].min()

    cols = ((patches_df[lon_col] - lon_min) / deg_lon).astype(int)
    rows = ((patches_df[lat_col] - lat_min) / deg_lat).astype(int)

    return "r" + rows.astype(str) + "_c" + cols.astype(str)


def spatial_kfold_split(
    patches_df: pd.DataFrame,
    lon_col: str = "lon_center",
    lat_col: str = "lat_center",
    cfg: SpatialKFoldConfig = SpatialKFoldConfig(),
) -> pd.DataFrame:
    """
    Devuelve una copia de `patches_df` con dos columnas nuevas:
      - block_id: id del bloque geográfico (20x20km por defecto)
      - fold: 0..n_folds-1 para folds de train/val en validación cruzada,
              o el string 'external_holdout' si cfg.held_out_external_fold
              es True (un bloque de bloques COMPLETAMENTE fuera de
              cualquier entrenamiento/tuning, reservado para la
              validación empírica final contra GeoBosques/GFW).

    Uso recomendado:
      - Reporta métricas (IoU/F1/Precision/Recall) promediadas sobre los
        `n_folds` folds de validación cruzada -> sección "Resultados del
        modelo" del artículo.
      - Usa 'external_holdout' UNA sola vez, al final, para el resultado
        headline de validación empírica -> tu aporte diferenciador.
    """
    df = patches_df.copy()
    df["block_id"] = assign_block_ids(df, lon_col, lat_col, cfg.block_size_km)

    unique_blocks = np.array(df["block_id"].unique().tolist(), dtype=object)
    rng = np.random.RandomState(cfg.random_seed)
    rng.shuffle(unique_blocks)

    n_groups = cfg.n_folds + 1 if cfg.held_out_external_fold else cfg.n_folds
    block_groups = np.array_split(unique_blocks, n_groups)

    block_to_fold: dict[str, object] = {}
    for fold_idx, blocks in enumerate(block_groups):
        if cfg.held_out_external_fold and fold_idx == n_groups - 1:
            label: object = "external_holdout"
        else:
            label = fold_idx
        for b in blocks:
            block_to_fold[b] = label

    df["fold"] = df["block_id"].map(block_to_fold)
    return df


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Resumen rápido: nº de parches y nº de bloques por fold (para tu tabla de metodología)."""
    return (
        df.groupby("fold")
        .agg(n_patches=("block_id", "count"), n_blocks=("block_id", "nunique"))
        .reset_index()
        .sort_values("fold", key=lambda s: s.astype(str))
    )


if __name__ == "__main__":
    # Ejemplo mínimo con datos sintéticos, solo para verificar que el
    # algoritmo corre. Sustituye por tus metadatos reales de parches
    # (típicamente un CSV generado al exportar los tiles desde GEE).
    rng = np.random.RandomState(0)
    demo = pd.DataFrame({
        "patch_id": [f"p{i}" for i in range(500)],
        "lon_center": rng.uniform(-69.75, -69.30, 500),
        "lat_center": rng.uniform(-11.55, -10.85, 500),
    })
    split = spatial_kfold_split(demo)
    print(summarize_split(split))
