"""
refine_labels.py
================
Etiquetas v3 para tala selectiva a partir de las 15 bandas del parche + la
etiqueta débil original. Motivación (análisis 2026-10-11):
  - Solo ~19 % de los píxeles de la etiqueta débil muestran caída de NDFI entre
    2024 y 2025; el resto no tiene cambio visible en las entradas del modelo.
  - Las etiquetas débiles mezclan bordes de río, minería y deforestación total.

Regla (inspirada en Souza et al. y en la firma de tala selectiva: claros
pequeños y trochas dentro de bosque que sigue en pie):
  1. Bosque antes:   NDFI_t0 >= FOREST_T0
  2. Pérdida:        dNDFI <= -T, con T adaptativo al ruido del parche
                     (max(T_MIN, K * desviación robusta del dNDFI en bosque))
  3. Excluir píxeles junto a aperturas que ya existían en t0 (ríos, caminos,
     chacras): ahí el "cambio" suele ser corregistro o bordes de cauce.
     Excluir agua y su entorno (MNDWI > 0 en t1, dilatado) y bordes de río.
  4. Excluir aperturas grandes en t1 (NDFI_t1 < 0.5 en componentes > BIG_PX)
     y su entorno: deforestación total, minería, chacras.
  5. Componentes conexos pequeños (<= MAX_PX) o alargados (trochas), con el
     entorno mayormente boscoso en t1 (el dosel sigue en pie).
  6. Píxeles de la etiqueta débil con apoyo espectral más laxo (dNDFI <= -T/2)
     y que pasan los mismos filtros de contexto también se incluyen.
Parches con dNDFI demasiado ruidoso (neblina/nubes) se marcan como `noisy` para
excluirlos del entrenamiento.

Uso:
    python -m src.refine_labels                 # escribe data/processed/masks_v3
    python -m src.refine_labels --preview 20    # PNGs de control en outputs/refine_preview
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

IDX = {n: i for i, n in enumerate(["B2", "B3", "B4", "B8", "B11", "B12", "NDVI_t1", "NDFI_t0", "NDFI_t1",
                                    "dNDFI", "NPV_t1", "Soil_t1", "sigma_NIR_t0", "sigma_NIR_t1", "dNBR"])}

P = dict(
    FOREST_T0=0.80,     # NDFI de bosque intacto ~0.9-0.95 (mediana medida en el dataset)
    T_MIN=0.15,         # caída mínima de NDFI
    K_NOISE=5.0,        # o 5 desviaciones robustas del ruido del parche
    NOISY_STD=0.08,     # parche descartado si el ruido robusto supera esto
    WATER_BUF=4,        # px alrededor de agua excluidos
    BIG_PX=150,         # >1.5 ha abierta en t1 = deforestación/minería/chacra
    BIG_BUF=4,          # px alrededor de aperturas grandes excluidos
    OPEN0_T=0.70,       # NDFI_t0 < esto = ya abierto en t0 (río, camino, chacra, píxel mixto)
    OPEN0_BUF=3,        # px alrededor de aperturas previas excluidos (bordes, error de corregistro)
    MIN_PX=2,           # 1 píxel suelto = ruido
    MAX_PX=60,          # claro de tala/patio: <= 0.6 ha
    LINE_MAX_PX=400,    # trocha: componente alargado hasta 4 ha
    STRAIGHT_MIN=5.0,   # sqrt(l1/l2) >= 5 => rasgo recto (trocha/camino)
    GROUP_BUF=6,        # fragmentos a <= 120 m se agrupan
    LONG_SPAN=40,       # grupos > 400 m de largo...
    CHAIN_MIN=2.5,      # ...con forma de cadena (no racimo compacto)
    ROAD_MIN=12.0,      # ...y no rectos como camino => quebrada/cauce, se descartan
    CTX_WIN=31,         # ventana de contexto (310 m)
    CTX_FOREST=0.70,    # fracción de bosque en t1 alrededor
)


def _shape(m: np.ndarray) -> tuple[float, float]:
    """Extensión a lo largo del eje principal (px) y rectitud sqrt(l1/l2) por PCA."""
    yy, xx = np.nonzero(m)
    if len(yy) < 3:
        return float(len(yy)), 1.0
    c = np.cov(np.vstack([yy, xx]).astype("float64"))
    l2, l1 = np.linalg.eigvalsh(c)
    v = np.linalg.eigh(c)[1][:, 1]
    proj = yy * v[0] + xx * v[1]
    return float(proj.max() - proj.min()), float(np.sqrt(l1 / max(l2, 1e-6)))


def refine(a: np.ndarray, weak: np.ndarray, p: dict = P) -> tuple[np.ndarray, dict]:
    t0, t1, dn = a[IDX["NDFI_t0"]], a[IDX["NDFI_t1"]], a[IDX["dNDFI"]]
    b3, b11 = a[IDX["B3"]], a[IDX["B11"]]
    valid = np.isfinite(dn) & np.isfinite(t0) & np.isfinite(t1)
    t0, t1, dn = np.nan_to_num(t0), np.nan_to_num(t1), np.nan_to_num(dn)

    forest0 = valid & (t0 >= p["FOREST_T0"])
    open0 = ndi.binary_dilation(valid & (t0 < p["OPEN0_T"]), iterations=p["OPEN0_BUF"])
    # ruido medido solo en bosque interior (lejos de cambios reales grandes como minería)
    interior = ndi.uniform_filter(forest0.astype("float32"), size=p["CTX_WIN"]) >= 0.95
    v = dn[forest0 & interior] if (forest0 & interior).sum() > 2000 else dn[forest0]
    if v.size > 500:
        q1, q3 = np.percentile(v, [25, 75])
        noise = (q3 - q1) / 1.349
    else:
        noise = np.inf
    info = {"noise": float(noise), "noisy": bool(noise > p["NOISY_STD"]), "forest0_frac": float(forest0.mean())}
    if info["noisy"]:
        info["pos_frac_v3"] = 0.0
        return np.zeros_like(weak, dtype=np.uint8), info
    T = max(p["T_MIN"], p["K_NOISE"] * noise)
    info["T"] = float(T)

    mndwi = (b3 - b11) / (b3 + b11 + 1e-6)
    water = ndi.binary_dilation(np.nan_to_num(mndwi) > 0, iterations=p["WATER_BUF"])

    open1 = valid & (t1 < 0.5)
    lab, n = ndi.label(open1, structure=np.ones((3, 3)))
    big = np.zeros_like(open1)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        big = np.isin(lab, np.nonzero(sizes > p["BIG_PX"])[0])
    big = ndi.binary_dilation(big, iterations=p["BIG_BUF"])

    ctx = ndi.uniform_filter((valid & (t1 >= 0.75)).astype("float32"), size=p["CTX_WIN"])
    allowed = forest0 & ~open0 & ~water & ~big & (ctx >= p["CTX_FOREST"])

    cand = allowed & (dn <= -T)
    cand |= allowed & (weak > 0) & (dn <= -T / 2)

    # Agrupa fragmentos cercanos (GROUP_BUF px) para reconocer rasgos lineales largos:
    # sinuosos = quebrada/cauce que se destapa (se descartan); rectos = trocha/camino (se conservan).
    groups, ng = ndi.label(ndi.binary_dilation(cand, iterations=p["GROUP_BUF"]), structure=np.ones((3, 3)))
    drop = np.zeros_like(cand)
    for g, sl in enumerate(ndi.find_objects(groups), start=1):
        if sl is None:
            continue
        gm = (groups[sl] == g) & cand[sl]
        span, straight = _shape(gm)
        # cadena larga ni compacta (racimo de claros, straight~1-2) ni recta (camino, >>12) => cauce
        if span > p["LONG_SPAN"] and p["CHAIN_MIN"] <= straight < p["ROAD_MIN"]:
            drop[sl] |= gm
    cand &= ~drop

    lab, n = ndi.label(cand, structure=np.ones((3, 3)))
    out = np.zeros_like(cand)
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        area = comp.sum()
        if area < p["MIN_PX"]:
            continue
        span, straight = _shape(comp)
        if area <= p["MAX_PX"] or (area <= p["LINE_MAX_PX"] and straight >= p["STRAIGHT_MIN"]):
            out[sl] |= comp
    info["pos_frac_v3"] = float(out.mean())
    return out.astype(np.uint8), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/processed/images")
    ap.add_argument("--masks", default="data/processed/masks")
    ap.add_argument("--out", default="data/processed/masks_v3")
    ap.add_argument("--metadata", default="data/processed/patches_metadata.csv")
    ap.add_argument("--preview", type=int, default=0, help="solo genera N PNGs de control (no escribe máscaras)")
    ap.add_argument("--ids", nargs="*", help="patch_ids concretos para el preview")
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata)
    ids = a.ids or meta.patch_id.tolist()
    if a.preview:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        outd = Path("outputs/refine_preview"); outd.mkdir(parents=True, exist_ok=True)
        if not a.ids:
            ids = meta.sample(a.preview, random_state=1).patch_id.tolist()
        for pid in ids:
            with rasterio.open(f"{a.images}/{pid}.tif") as s:
                img = s.read().astype("float32")
            with rasterio.open(f"{a.masks}/{pid}.tif") as s:
                weak = s.read(1)
            m, info = refine(img, weak)
            def st(x):
                lo, hi = np.nanpercentile(x, [2, 98]); return np.clip((np.nan_to_num(x) - lo) / (hi - lo + 1e-9), 0, 1)
            rgb = np.dstack([st(img[IDX[b]]) for b in ("B4", "B3", "B2")])
            f, ax = plt.subplots(1, 3, figsize=(15, 5.3), dpi=80)
            ax[0].imshow(rgb); ax[0].set_title(f"{pid} RGB")
            ax[1].imshow(rgb); ax[1].contour(weak > 0, levels=[.5], colors="yellow", linewidths=.8); ax[1].set_title("débil (amarillo)")
            ax[2].imshow(rgb); ax[2].contour(m > 0, levels=[.5], colors="red", linewidths=.9)
            ax[2].set_title(f"v3 (rojo) ruido={info['noise']:.3f} {'RUIDOSO' if info['noisy'] else ''}")
            for x in ax: x.axis("off")
            plt.tight_layout(); plt.savefig(outd / f"{pid}.png"); plt.close()
        print(f"previews -> {outd}")
        return

    Path(a.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for pid in ids:
        with rasterio.open(f"{a.images}/{pid}.tif") as s:
            img = s.read().astype("float32")
        with rasterio.open(f"{a.masks}/{pid}.tif") as s:
            weak = s.read(1); prof = s.profile
        m, info = refine(img, weak)
        with rasterio.open(f"{a.out}/{pid}.tif", "w", **prof) as d:
            d.write(m[None])
        rows.append({"patch_id": pid, **info})
    info = pd.DataFrame(rows)
    meta = meta.drop(columns=[c for c in info.columns if c != "patch_id" and c in meta.columns]).merge(info, on="patch_id")
    meta.to_csv(a.metadata.replace(".csv", "_v3.csv"), index=False)
    print(meta.groupby("split_role").agg(n=("patch_id", "count"), noisy=("noisy", "sum"),
                                          pos_v3=("pos_frac_v3", "mean"), pos_weak=("pos_frac", "mean")))


if __name__ == "__main__":
    main()
