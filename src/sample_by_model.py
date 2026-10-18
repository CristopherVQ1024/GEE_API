"""Muestra nueva para fotointerpretar, estratificada por el puntaje del modelo v3.

Objetivo: conseguir más positivos para el test sin etiquetar al azar (daría muy pocos) y
sin sesgar todo hacia lo que el modelo ya ve. Se puntúan los parches aún sin etiquetar
(Tambopata con el modelo del fold fuera de muestra; Tahuamanu con el ensamble) y se muestrea:
  alto  : mayor puntaje top100 (más probables positivos),
  medio : puntaje intermedio,
  bajo  : puntaje bajo al azar (mide lo que se le escapa al modelo).
Cada estrato se reparte entre zonas (40 % Tahuamanu, que es la prueba de transferencia). Se ordena por frac05 (fracción de px > 0,5), porque top100 se satura en 1,0. Se excluyen los parches `noisy` (neblina: no se puede fotointerpretar).
La herramienta de etiquetado NO muestra el puntaje (para no sesgar la decisión).

    python -m src.sample_by_model --ckpt outputs/v3/checkpoints --out data/labeling/sample_r3.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate import get_device, load_config, load_fold_checkpoints
from src.evaluate_patch_level import predict, scores

LAB = Path("data/labeling")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v3/checkpoints")
    ap.add_argument("--metadata", default="data/processed/patches_metadata_v3.csv")
    ap.add_argument("--out", default="data/labeling/sample_r3.csv")
    ap.add_argument("--scores", default="outputs/patch_eval/scores_all_v3.csv", help="caché de puntajes")
    ap.add_argument("--n_alto", type=int, default=30)
    ap.add_argument("--n_medio", type=int, default=15)
    ap.add_argument("--n_bajo", type=int, default=15)
    ap.add_argument("--frac_tahuamanu", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    meta = pd.read_csv(a.metadata)
    hechos = set()
    for f in LAB.glob("sample*.csv"):
        if f.resolve() != Path(a.out).resolve():  # no excluir la propia salida: así es reproducible
            hechos |= set(pd.read_csv(f).patch_id)
    meta = meta[~meta.patch_id.isin(hechos) & ~meta.noisy.astype(bool)].copy()
    meta["zona"] = meta.patch_id.str.split("_").str[0]

    if Path(a.scores).exists():
        sc = pd.read_csv(a.scores)
    else:
        cfg, dev = load_config(), get_device()
        models = load_fold_checkpoints(a.ckpt, cfg["model"], dev)
        rows = []
        for r in meta.itertuples():
            path = f"{cfg['paths']['patches_images']}/{r.patch_id}.tif"
            if r.fold == "external_holdout":
                p = np.mean([predict(m, path, dev) for m in models], axis=0)
            else:
                p = predict(models[int(r.fold)], path, dev)
            rows.append({"patch_id": r.patch_id, **scores(p)})
        sc = pd.DataFrame(rows)
        Path(a.scores).parent.mkdir(parents=True, exist_ok=True)
        sc.to_csv(a.scores, index=False)
    d = meta.merge(sc, on="patch_id")
    print(f"candidatos sin etiquetar: {len(d)}  ({d.zona.value_counts().to_dict()})")

    rng = np.random.default_rng(a.seed)
    out = []
    # Tahuamanu (zona no vista) tiene pocos candidatos y es la prueba de transferencia: cuota fija mayor.
    cuota = {"tahuamanu": a.frac_tahuamanu, "tambopata": 1 - a.frac_tahuamanu}
    d["orden"] = d.frac05  # top100 se satura en 1,0 en Tambopata; frac05 separa mejor
    for estrato, n in (("alto", a.n_alto), ("medio", a.n_medio), ("bajo", a.n_bajo)):
        for z, g in d.groupby("zona"):
            k = max(1, round(n * cuota[z]))
            if estrato == "alto":
                pick = g.sort_values("orden", ascending=False).head(k)
            else:
                q = g.orden.rank(pct=True, method="first")
                pool = g[(q >= 0.4) & (q < 0.8)] if estrato == "medio" else g[q < 0.4]
                pick = pool.sample(min(k, len(pool)), random_state=int(rng.integers(1e9)))
            out.append(pick.assign(estrato=estrato))
    s = pd.concat(out).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    s = s.rename(columns={"frac05": "score_modelo"})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s[["patch_id", "zona", "estrato", "pos_frac", "fold", "score_modelo"]].to_csv(a.out, index=False)
    print(s.groupby(["estrato", "zona"]).score_modelo.agg(["count", "mean", "min", "max"]).round(3))
    print(f"-> {a.out} ({len(s)} parches)")


if __name__ == "__main__":
    main()
