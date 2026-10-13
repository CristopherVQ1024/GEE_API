"""Evaluación a nivel de PARCHE contra la fotointerpretación (data/labeling).

Etiqueta final = la de Cristopher (labels.csv); si el parche no fue revisado, la
etiqueta de Claude (labels_claude.csv + labels_claude_r2.csv). Positivo = 'si'; 'dudoso' se excluye;
'no' y 'otro' son negativos (también se reporta excluyendo 'otro').

Para evitar fuga, los parches de Tambopata se puntúan solo con el modelo del fold
al que pertenecen (predicción fuera de muestra); Tahuamanu, con el ensamble.

    python -m src.evaluate_patch_level --ckpt outputs/checkpoints --tag base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluate import get_device, load_config, load_fold_checkpoints

LAB = Path("data/labeling")


def final_labels() -> pd.DataFrame:
    c = pd.concat([pd.read_csv(LAB / f)[["patch_id", "label"]] for f in ("labels_claude.csv", "labels_claude_r2.csv")
                   if (LAB / f).exists()]).assign(origen="claude")
    u = pd.read_csv(LAB / "labels.csv")[["patch_id", "label"]].assign(origen="usuario")
    return pd.concat([u, c[~c.patch_id.isin(u.patch_id)]]).reset_index(drop=True)


def predict(model_d, img_path, device):
    img = rasterio.open(img_path).read().astype("float32")
    m = np.array(model_d["norm_stats"]["mean"], dtype="float32").reshape(-1, 1, 1)
    s = np.array(model_d["norm_stats"]["std"], dtype="float32").reshape(-1, 1, 1)
    x = np.nan_to_num((img - m) / np.clip(s, 1e-6, None), nan=0.0)
    with torch.no_grad():
        p = torch.sigmoid(model_d["model"](torch.from_numpy(x).unsqueeze(0).to(device)))
    return p[0, 0].cpu().numpy()


def scores(p):
    f = np.sort(p.ravel())[::-1]
    return {"max": f[0], "top100": f[:100].mean(), "mean": p.mean(), "frac05": (p > 0.5).mean()}


def auc_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(s); v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            v.append(roc_auc_score(y[i], s[i]))
    return np.percentile(v, [2.5, 97.5]).round(2).tolist() if v else [None, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--metadata", default="data/processed/patches_metadata.csv")
    a = ap.parse_args()
    cfg, dev = load_config(), get_device()
    models = load_fold_checkpoints(a.ckpt, cfg["model"], dev)
    meta = pd.read_csv(a.metadata).set_index("patch_id")
    lab = final_labels()
    rows = []
    for pid in lab.patch_id:
        path = f"{cfg['paths']['patches_images']}/{pid}.tif"
        fold = meta.loc[pid, "fold"]
        if fold == "external_holdout":
            ps = np.mean([predict(m, path, dev) for m in models], axis=0)
        else:
            ps = predict(models[int(fold)], path, dev)  # fold k <-> unet_fold{k}_best.pt (orden alfabético)
        rows.append({"patch_id": pid, **scores(ps), "weak_pos_frac": meta.loc[pid, "pos_frac"]})
    d = lab.merge(pd.DataFrame(rows), on="patch_id")
    d["zona"] = d.patch_id.str.split("_").str[0]
    Path("outputs/patch_eval").mkdir(parents=True, exist_ok=True)
    d.to_csv(f"outputs/patch_eval/scores_{a.tag}.csv", index=False)

    out = {}
    for nombre, filt in [("todo(no+otro)", lambda x: x.label != "dudoso"),
                         ("sin_otro", lambda x: x.label.isin(["si", "no"]))]:
        for zona in ["todas", "tambopata", "tahuamanu"]:
            x = d[filt(d)]
            x = x if zona == "todas" else x[x.zona == zona]
            y = (x.label == "si").astype(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            for sc in ["max", "top100", "mean", "frac05", "weak_pos_frac"]:
                out[f"{nombre}|{zona}|{sc}"] = {
                    "n": len(x), "pos": int(y.sum()),
                    "auc": round(roc_auc_score(y, x[sc]), 3), "auc_ic95": auc_ci(y, x[sc]),
                    "ap": round(average_precision_score(y, x[sc]), 3), "base": round(y.mean(), 3)}
    json.dump(out, open(f"outputs/patch_eval/metrics_{a.tag}.json", "w"), indent=1)
    print(pd.DataFrame(out).T.to_string())


if __name__ == "__main__":
    main()
