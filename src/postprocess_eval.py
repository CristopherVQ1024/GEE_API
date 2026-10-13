"""¿Mejora el postproceso la detección a nivel de parche? (modelo v3 vs fotointerpretación)

Compara, sobre los parches fotointerpretados (misma referencia que evaluate_patch_level):
  - TTA (promedio de flips) frente a predicción simple;
  - filtro por tamaño de componente (quita manchas minúsculas y aperturas grandes);
  - filtro de forma (quita trazos largos: caminos/quebradas);
  - filtro de neblina (parches con ruido de fondo alto en dNDFI);
  - combinaciones.
Tambopata se puntúa con el modelo del fold fuera de muestra; Tahuamanu con el ensamble.

    python -m src.postprocess_eval --ckpt outputs/v3/checkpoints --cache <ruta>.npz
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset import BAND_NAMES
from src.evaluate import get_device, load_config, load_fold_checkpoints
from src.evaluate_patch_level import final_labels, predict
from src.refine_labels import _shape

S8 = np.ones((3, 3))
DNDFI = BAND_NAMES.index("dNDFI")


def predict_tta(model_d, img_path, device):
    """Promedio de 4 vistas (original, flip horizontal, vertical, ambos). Los parches son 256x257: sin rot90."""
    img = rasterio.open(img_path).read().astype("float32")
    m = np.array(model_d["norm_stats"]["mean"], dtype="float32").reshape(-1, 1, 1)
    s = np.array(model_d["norm_stats"]["std"], dtype="float32").reshape(-1, 1, 1)
    x = torch.from_numpy(np.nan_to_num((img - m) / np.clip(s, 1e-6, None), nan=0.0)).unsqueeze(0).to(device)
    out = 0
    with torch.no_grad():
        for dims in ([], [3], [2], [2, 3]):
            xi = x.flip(dims) if dims else x
            pi = torch.sigmoid(model_d["model"](xi))
            out = out + (pi.flip(dims) if dims else pi)
    return (out / 4)[0, 0].cpu().numpy()


def build_cache(args, cfg, dev, lab, meta):
    models = load_fold_checkpoints(args.ckpt, cfg["model"], dev)
    probs, probs_tta, dndfi = {}, {}, {}
    for pid in lab.patch_id:
        path = f"{cfg['paths']['patches_images']}/{pid}.tif"
        fold = meta.loc[pid, "fold"]
        ms = models if fold == "external_holdout" else [models[int(fold)]]
        probs[pid] = np.mean([predict(m, path, dev) for m in ms], axis=0)
        probs_tta[pid] = np.mean([predict_tta(m, path, dev) for m in ms], axis=0)
        dndfi[pid] = rasterio.open(path).read(DNDFI + 1).astype("float32")
    np.savez_compressed(args.cache, **{f"p|{k}": v for k, v in probs.items()},
                        **{f"t|{k}": v for k, v in probs_tta.items()}, **{f"d|{k}": v for k, v in dndfi.items()})


def components(p, thr):
    lab, n = ndi.label(p > thr, structure=S8)
    if n == 0:
        return lab, np.array([]), []
    idx = np.arange(1, n + 1)
    areas = ndi.sum(np.ones_like(p), lab, idx).astype(int)
    shapes = [_shape(lab == i) for i in idx]  # (extensión px, rectitud)
    return lab, areas, shapes


def filtered_map(p, thr, amin=0, amax=10**9, emax=10**9):
    lab, areas, shapes = components(p, thr)
    keep = np.zeros(p.shape, bool)
    for i, (a, (ext, _)) in enumerate(zip(areas, shapes), start=1):
        if amin <= a <= amax and ext <= emax:
            keep |= lab == i
    return np.where(keep, p, 0.0)


def score(p):
    f = np.sort(p.ravel())[::-1]
    return f[:100].mean()


def auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5])


def paired_delta(y, s1, s0, n=2000, seed=0):
    """IC95 bootstrap de AUC(s1) - AUC(s0) (mismos remuestreos)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], np.asarray(s1)[i]) - roc_auc_score(y[i], np.asarray(s0)[i]))
    return np.percentile(d, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v3/checkpoints")
    ap.add_argument("--metadata", default="data/processed/patches_metadata_v3.csv")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="outputs/patch_eval/postprocess_v3.csv")
    a = ap.parse_args()
    cfg, dev = load_config(), get_device()
    meta = pd.read_csv(a.metadata).set_index("patch_id")
    lab = final_labels()
    lab = lab[lab.label != "dudoso"].reset_index(drop=True)
    if not Path(a.cache).exists():
        build_cache(a, cfg, dev, lab, meta)
    z = np.load(a.cache)
    P = {k: z[f"p|{k}"] for k in lab.patch_id}
    T = {k: z[f"t|{k}"] for k in lab.patch_id}
    D = {k: z[f"d|{k}"] for k in lab.patch_id}
    y = (lab.label == "si").astype(int).values
    sin_otro = lab.label.isin(["si", "no"]).values
    noise = meta.loc[lab.patch_id, "noise"].values
    # ruido de fondo propio: desviación robusta (MAD) de dNDFI en el parche
    mad = np.array([1.4826 * np.nanmedian(np.abs(D[k] - np.nanmedian(D[k]))) for k in lab.patch_id])
    # sesgo de fondo: mediana de dNDFI (neblina desplaza todo el parche)
    bias = np.array([np.nanmedian(D[k]) for k in lab.patch_id])

    variants = {}
    for name, M in (("simple", P), ("TTA", T)):
        variants[f"{name}|sin filtro"] = [score(M[k]) for k in lab.patch_id]
        for amin, amax in itertools.product([4, 8, 15], [10**9, 300, 150]):
            variants[f"{name}|tamaño [{amin},{'inf' if amax > 10**8 else amax}]"] = [
                score(filtered_map(M[k], 0.5, amin, amax)) for k in lab.patch_id]
        for emax in (25, 40, 60):
            variants[f"{name}|extensión<={emax}px"] = [
                score(filtered_map(M[k], 0.5, emax=emax)) for k in lab.patch_id]
        for amin, emax in itertools.product([4, 8], [40, 60]):
            variants[f"{name}|tamaño>={amin} + ext<={emax}"] = [
                score(filtered_map(M[k], 0.5, amin=amin, emax=emax)) for k in lab.patch_id]
    # neblina: se desplaza el puntaje de los parches ruidosos
    base = np.array(variants["simple|sin filtro"])
    for q in (75, 85, 90):
        t = np.percentile(noise, q)
        variants[f"simple|neblina: ruido>p{q} -> 0"] = np.where(noise > t, 0, base)
        variants[f"simple|tamaño>=8 + neblina p{q}"] = np.where(noise > t, 0, variants["simple|tamaño [8,inf]"])
    variants["simple|score/(ruido)"] = base / np.clip(noise, 1e-3, None)
    variants["simple|score*(1-ruido/max)"] = base * (1 - noise / noise.max())
    variants["_ref: solo ruido"] = noise
    variants["_ref: solo MAD dNDFI"] = mad
    variants["_ref: -|sesgo dNDFI|"] = -np.abs(bias)

    rows = []
    for nombre, s in variants.items():
        s = np.asarray(s, float)
        r = {"variante": nombre}
        for tag, mask in (("todo", np.ones(len(y), bool)), ("sin_otro", sin_otro)):
            yy, ss = y[mask], s[mask]
            r[f"auc_{tag}"] = roc_auc_score(yy, ss)
            r[f"ap_{tag}"] = average_precision_score(yy, ss)
        lo, hi = auc_ci(y, s)
        r["ic95_todo"] = f"[{lo:.2f},{hi:.2f}]"
        d0 = paired_delta(y, s, base)
        r["dAUC_vs_simple"] = f"{r['auc_todo'] - roc_auc_score(y, base):+.3f} [{d0[0]:+.2f},{d0[1]:+.2f}]"
        rows.append(r)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        print(f"n={len(y)} positivos={y.sum()} sin_otro n={sin_otro.sum()}")
        print(df.sort_values("auc_todo", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
