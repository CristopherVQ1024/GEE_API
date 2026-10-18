"""CSV combinado para entrenar el modelo v4: Madre de Dios (etiquetas v3 con incierto = 255, masks_v3i) + DETER (etiquetas v4).

- Tambopata conserva sus folds espaciales (0–4); Tahuamanu sigue como `external_holdout`.
- Los bloques DETER se reparten en los mismos 5 folds por ubicación (un mismo bloque de 10 km de
  años distintos cae en el mismo fold, para que validación y entrenamiento no compartan lugar).
- Se excluyen parches ruidosos (neblina) y con < 80 % de píxeles válidos.
El benchmark DETER de prueba (test2026) NO entra aquí.

    python -m src.build_v4_metadata --deter data/deter/train2025 data/deter/train2024
"""
from __future__ import annotations

import argparse
import hashlib

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdd", default="data/processed/patches_metadata_v3.csv")
    ap.add_argument("--deter", nargs="+", required=True)
    ap.add_argument("--out", default="data/processed/patches_metadata_v4.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    a = ap.parse_args()

    m = pd.read_csv(a.mdd)
    m = m[~m.noisy.astype(bool)].copy()
    m["image_path"] = "data/processed/images/" + m.patch_id + ".tif"
    m["mask_path"] = "data/processed/masks_v3i/" + m.patch_id + ".tif"
    m["source"] = "mdd"
    parts = [m[["patch_id", "lon_center", "lat_center", "fold", "image_path", "mask_path", "source"]]]
    for d in a.deter:
        t = pd.read_csv(f"{d}/metadata.csv")
        t = t[(~t.noisy.astype(bool)) & (t.valid_frac >= 0.8)].copy()
        t["image_path"] = f"{d}/images/" + t.patch_id + ".tif"
        t["mask_path"] = f"{d}/labels_v4/" + t.patch_id + ".tif"
        # fold estable por ubicación del bloque (epsg_bx_by)
        t["fold"] = t.block_id.map(lambda b: int(hashlib.md5(b.encode()).hexdigest(), 16) % a.n_folds).astype(str)
        t["source"] = "deter_" + t.uf
        parts.append(t[["patch_id", "lon_center", "lat_center", "fold", "image_path", "mask_path", "source"]])
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False)
    print(out.groupby(["fold", out.source.str[:5]]).size().unstack(fill_value=0))
    print(f"{len(out)} parches -> {a.out}")


if __name__ == "__main__":
    main()
