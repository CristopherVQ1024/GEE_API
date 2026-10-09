"""
prepare_dataset.py
==================
Reproduce localmente la "celda 6" del notebook v2: toma los exports crudos de
GEE (17 bandas: 15 features + label + valid) y genera el layout que espera
src/dataset.py:
    data/processed/images/<patch_id>.tif   (15 bandas)
    data/processed/masks/<patch_id>.tif    (1 banda, label)
    data/processed/patches_metadata_raw.csv (patch_id, lon_center, lat_center,
                                             split_role, pos_frac, invalid_frac)
Uso: python -m src.prepare_dataset --src dataset
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


def main(src: str, out: str) -> None:
    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    to_ll = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    rows = []
    for p in sorted(Path(src).glob("*.tif")):
        with rasterio.open(p) as s:
            arr = s.read()
            prof = s.profile
            b = s.bounds
        assert arr.shape[0] == 17, f"{p.name}: {arr.shape[0]} bandas (se esperaban 17)"
        label = np.nan_to_num(arr[15], nan=0.0)
        valid = np.nan_to_num(arr[16], nan=0.0)
        pid = p.stem
        prof.update(count=15, dtype="float32")
        with rasterio.open(out / "images" / p.name, "w", **prof) as d:
            d.write(arr[:15].astype("float32"))
        prof.update(count=1, dtype="uint8", nodata=None)
        with rasterio.open(out / "masks" / p.name, "w", **prof) as d:
            d.write((label > 0.5).astype("uint8")[None])
        lon, lat = to_ll.transform((b.left + b.right) / 2, (b.bottom + b.top) / 2)
        rows.append(dict(
            patch_id=pid, lon_center=lon, lat_center=lat,
            split_role="train" if pid.startswith("tambopata") else "test_transfer",
            pos_frac=float((label > 0.5).mean()), invalid_frac=float((valid < 0.5).mean()),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out / "patches_metadata_raw.csv", index=False)
    print(df.groupby("split_role").agg(n=("patch_id", "count"), pos=("pos_frac", "mean"), inv=("invalid_frac", "max")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="dataset")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    main(a.src, a.out)
