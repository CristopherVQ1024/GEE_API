"""Etiquetas v4 para parches DETER: regla espectral v3 confirmada por polígonos oficiales.

  1 = DENTRO de un polígono de tala selectiva DETER de la ventana (dilatado 30 m): bosque en t0 con
      caída de NDFI >= T/2 (T = umbral adaptativo de la regla v3), fuera de agua y de aperturas
      gruesas (> ~50 m de ancho). A diferencia de v3, se aceptan caminos de extracción sinuosos:
      DETER confirma que ahí hubo tala, así que no son quebradas;
  0 = resto, incluidos los cambios dentro de polígonos DETER de deforestación/minería/quema
      (enseñan que una apertura total NO es tala selectiva);
255 = incierto (no cuenta en la pérdida): cambios que la regla v3 marca —o descarta por forma—
      fuera de todo polígono DETER (DETER no mapea todo) y la tala DETER de otras fechas.
Parches con dNDFI ruidoso (neblina) se marcan `noisy`.

    python -m src.deter_labels --data data/deter/train2025
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

from src.refine_labels import IDX, P, refine

IGNORE = 255


def v4_label(img, m, p=P):
    """Etiqueta v4 de un parche DETER (ver docstring del módulo)."""
    cs, cs_other, other = m[0] > 0, m[1] > 0, m[2] > 0
    rule, info, uncertain = refine(img, np.zeros(cs.shape, np.uint8), return_uncertain=True)
    lab = np.zeros(cs.shape, np.uint8)
    if info["noisy"]:
        info.update(pos_frac_v4=0.0, ign_frac_v4=0.0)
        return lab, info
    T = info["T"]
    t0, t1, dn = (np.nan_to_num(img[IDX[k]]) for k in ("NDFI_t0", "NDFI_t1", "dNDFI"))
    valid = np.isfinite(img[IDX["dNDFI"]])
    b3, b11 = img[IDX["B3"]], img[IDX["B11"]]
    water = ndi.binary_dilation(np.nan_to_num((b3 - b11) / (b3 + b11 + 1e-6)) > 0, iterations=p["WATER_BUF"])
    forest0 = valid & (t0 >= p["FOREST_T0"])
    # aperturas gruesas en t1 (> ~50 m de ancho: minería, chacra, tala rasa); los caminos de
    # extracción (10–30 m) desaparecen con la erosión y se conservan
    thick = ndi.binary_dilation(ndi.binary_erosion(valid & (t1 < 0.5), iterations=2), iterations=5)
    ctx = ndi.uniform_filter((valid & (t1 >= 0.75)).astype("float32"), size=p["CTX_WIN"])
    cs_d = ndi.binary_dilation(cs, iterations=3)
    inside = cs_d & forest0 & ~water & ~thick & (ctx >= 0.5) & (dn <= -T / 2)
    lab_i, _ = ndi.label(inside, structure=np.ones((3, 3)))
    sizes = np.bincount(lab_i.ravel()); sizes[0] = 0
    inside = np.isin(lab_i, np.nonzero(sizes >= p["MIN_PX"])[0])

    outside_change = (rule > 0) | uncertain
    lab[outside_change & ~cs_d & ~other] = IGNORE          # cambio sin polígono: DETER no mapea todo
    lab[ndi.binary_dilation(cs_other, iterations=3) & ~cs_d] = IGNORE
    lab[inside] = 1
    info["pos_frac_v4"] = float((lab == 1).mean())
    info["ign_frac_v4"] = float((lab == IGNORE).mean())
    return lab, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    d = Path(a.data)
    meta = pd.read_csv(d / "metadata.csv")
    (d / "labels_v4").mkdir(exist_ok=True)
    rows = []
    for pid in meta.patch_id:
        with rasterio.open(d / "images" / f"{pid}.tif") as s:
            img = s.read().astype(np.float32)
        with rasterio.open(d / "masks" / f"{pid}.tif") as s:
            m = s.read(); prof = s.profile
        lab, info = v4_label(img, m)
        prof.update(count=1, dtype="uint8")
        with rasterio.open(d / "labels_v4" / f"{pid}.tif", "w", **prof) as dst:
            dst.write(lab[None])
        rows.append({"patch_id": pid, **info})
    info = pd.DataFrame(rows)
    meta = meta.drop(columns=[c for c in info.columns if c != "patch_id" and c in meta.columns]).merge(info, on="patch_id")
    meta.to_csv(d / "metadata.csv", index=False)
    print(meta.groupby("uf").agg(n=("patch_id", "size"), noisy=("noisy", "sum"), pos=("pos_frac_v4", "mean"),
                                 ign=("ign_frac_v4", "mean"), cs=("cs_frac", "mean")).round(4).to_string())


if __name__ == "__main__":
    main()
