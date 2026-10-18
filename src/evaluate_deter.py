"""Evalúa un modelo contra los polígonos oficiales de tala selectiva de INPE DETER.

Conjunto: salida de src/deter_dataset.py (máscaras de 3 bandas: 1 = tala selectiva DETER en la
ventana del par, 2 = tala selectiva DETER de otras fechas, 3 = otras clases DETER en el periodo).
Métricas:
  - Parche: AUC/AP separando parches con tala DETER (cs_frac >= --min_cs) de parches sin ningún
    evento DETER (las tres bandas en 0). Puntaje = media de los 100 píxeles más probables (top100)
    y fracción de píxeles > umbral (frac).
  - Polígono: % de polígonos DETER (componentes conexos, >= 1 ha) cuya densidad de detección supera
    el percentil 95 de la densidad en parches sin eventos DETER (azar = 5 %).
  - Píxel: densidad de detección dentro de tala DETER / densidad en bosque sin eventos ("lift"),
    y dentro de otras clases DETER (¿confunde con deforestación/minería?).
  - Bloque: correlación de Spearman entre área detectada y área DETER por bloque.

    python -m src.evaluate_deter --data data/deter/test2026 --ckpt outputs/v3/checkpoints --tag v3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import get_device, load_config, load_fold_checkpoints


def predict_ensemble(models, img, device, tta=False):
    probs = []
    for m in models:
        mean = np.array(m["norm_stats"]["mean"], np.float32).reshape(-1, 1, 1)
        std = np.array(m["norm_stats"]["std"], np.float32).reshape(-1, 1, 1)
        x = torch.from_numpy(np.nan_to_num((img - mean) / np.clip(std, 1e-6, None), nan=0.0)).unsqueeze(0).to(device)
        with torch.no_grad():
            views = ([], [3], [2], [2, 3]) if tta else ([],)
            p = 0
            for dims in views:
                xi = x.flip(dims) if dims else x
                pi = torch.sigmoid(m["model"](xi))
                p = p + (pi.flip(dims) if dims else pi)
            probs.append((p / len(views))[0, 0].cpu().numpy())
    return np.mean(probs, 0)


def auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/deter/test2026")
    ap.add_argument("--ckpt", default="outputs/v3/checkpoints")
    ap.add_argument("--tag", default="v3")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min_cs", type=float, default=0.02, help="fracción mínima de tala DETER para parche positivo")
    ap.add_argument("--min_valid", type=float, default=0.8)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    if (data / "metadata.csv").exists():
        meta = pd.read_csv(data / "metadata.csv")
    else:  # conjunto aún descargándose: usa los bloques terminados
        meta = pd.DataFrame([r for f in sorted((data / "blocks_done").glob("*.json")) for r in json.loads(f.read_text())])
        print(f"(parcial: {meta.block_id.nunique()} bloques)")
    meta = meta[meta.valid_frac >= a.min_valid].reset_index(drop=True)
    cfg, dev = load_config(), get_device()
    models = load_fold_checkpoints(a.ckpt, cfg["model"], dev)
    out_dir = Path("outputs/deter_eval") / data.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, polys = [], []
    agg = {k: [0, 0] for k in ("cs", "cs_other", "other", "clean_forest")}  # [detectados, total]
    for r in meta.itertuples():
        with rasterio.open(data / "images" / f"{r.patch_id}.tif") as s:
            img = s.read().astype(np.float32)
        with rasterio.open(data / "masks" / f"{r.patch_id}.tif") as s:
            m = s.read()
        p = predict_ensemble(models, img, dev, a.tta)
        det = p > a.thr
        valid = np.isfinite(img[9])
        forest0 = valid & (np.nan_to_num(img[7]) >= 0.8)  # bosque en t0 (NDFI)
        any_deter = (m > 0).any(0)
        zones = {"cs": m[0] > 0, "cs_other": (m[1] > 0) & ~(m[0] > 0), "other": (m[2] > 0) & ~(m[0] > 0),
                 "clean_forest": forest0 & ~ndi.binary_dilation(any_deter, iterations=10)}
        for k, z in zones.items():
            z = z & valid
            agg[k][0] += int((det & z).sum()); agg[k][1] += int(z.sum())
        f = np.sort(p[valid].ravel())[::-1] if valid.any() else np.zeros(1)
        rows.append({"patch_id": r.patch_id, "block_id": r.block_id, "uf": r.uf, "cs_frac": r.cs_frac,
                     "clean": bool(not any_deter.any()),
                     "other_only": bool((m[2] > 0).mean() >= a.min_cs and not (m[0] > 0).any() and not (m[1] > 0).any()), "top100": float(f[:100].mean()),
                     "frac": float(det[valid].mean()) if valid.any() else 0.0, "det_px": int(det.sum())})
        lab, n = ndi.label(m[0] > 0)
        for i, sl in enumerate(ndi.find_objects(lab), start=1):
            comp = (lab[sl] == i)
            if comp.sum() >= 100:  # >= 1 ha dentro del parche
                polys.append({"patch_id": r.patch_id, "uf": r.uf, "area_px": int(comp.sum()),
                              "det_px": int((det[sl] & comp).sum())})
    d = pd.DataFrame(rows); P = pd.DataFrame(polys)
    d.to_csv(out_dir / f"patches_{a.tag}.csv", index=False)

    res = {"n_patches": len(d), "tag": a.tag, "ckpt": a.ckpt, "thr": a.thr, "tta": a.tta}
    x = d[(d.cs_frac >= a.min_cs) | d.clean]
    y = (x.cs_frac >= a.min_cs).astype(int)
    res["patch"] = {"n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    for sc in ("top100", "frac"):
        res["patch"][sc] = {"auc": round(roc_auc_score(y, x[sc]), 3), "auc_ic95": auc_ci(y, x[sc]),
                            "ap": round(average_precision_score(y, x[sc]), 3), "base": round(float(y.mean()), 3)}
    # negativos difíciles: parches con deforestación/minería/quema DETER y sin tala selectiva
    xh = d[(d.cs_frac >= a.min_cs) | d.other_only]
    yh = (xh.cs_frac >= a.min_cs).astype(int)
    if yh.nunique() == 2:
        res["patch_hard_neg"] = {"n_pos": int(yh.sum()), "n_neg": int((1 - yh).sum()),
                                 "auc_top100": round(roc_auc_score(yh, xh.top100), 3), "auc_ic95": auc_ci(yh, xh.top100),
                                 "auc_frac": round(roc_auc_score(yh, xh.frac), 3)}
    res["patch_by_uf"] = {}
    for uf, g in x.groupby("uf"):
        yy = (g.cs_frac >= a.min_cs).astype(int)
        if yy.nunique() == 2:
            res["patch_by_uf"][uf] = {"n_pos": int(yy.sum()), "n_neg": int((1 - yy).sum()),
                                      "auc_top100": round(roc_auc_score(yy, g.top100), 3)}
    dens = {k: v[0] / max(v[1], 1) for k, v in agg.items()}
    res["pixel"] = {"det_density": {k: round(v, 5) for k, v in dens.items()},
                    "lift_cs_vs_clean_forest": round(dens["cs"] / max(dens["clean_forest"], 1e-9), 2),
                    "lift_other_vs_clean_forest": round(dens["other"] / max(dens["clean_forest"], 1e-9), 2),
                    "share_of_detections_in_cs": round(agg["cs"][0] / max(sum(v[0] for v in agg.values()), 1), 3)}
    if len(P):
        # Un polígono se considera detectado si su densidad de detecciones supera el percentil 95 de
        # la densidad en parches sin ningún evento DETER (azar = 5 % por construcción).
        thr_d = float(np.percentile(d[d.clean].frac, 95)) if d.clean.any() else 0.0
        P["density"] = P.det_px / P.area_px
        hit = P.density > thr_d
        res["polygon"] = {"n": len(P), "clean_density_p95": round(thr_d, 5), "hit_rate": round(float(hit.mean()), 3),
                          "chance_hit_rate": 0.05, "median_density_ratio": round(float(P.density.median() / max(dens["clean_forest"], 1e-9)), 2),
                          "hit_rate_by_uf": P.assign(h=hit).groupby("uf").h.mean().round(3).to_dict()}
    b = d.groupby("block_id").agg(det=("det_px", "sum"), cs=("cs_frac", "mean"))
    rho = spearmanr(b.det, b.cs)
    res["block_spearman_det_vs_deter_area"] = {"rho": round(float(rho.statistic), 3), "p": round(float(rho.pvalue), 4), "n": len(b)}
    json.dump(res, open(out_dir / f"metrics_{a.tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
