"""Resultados de la muestra r3 (etiquetada por Claude, a ciegas) por estrato de puntaje del modelo v3.

La muestra r3 NO es aleatoria: «alto» son los candidatos del modelo. Por eso se reporta la
proporción de «sí» por estrato (precisión de los candidatos y positivos que se le escapan en «bajo»)
y el AUC conjunto con r1+r2 solo como referencia (sesgado).

    python -m src.evaluate_r3
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

LAB = "data/labeling"


def main() -> None:
    s = pd.read_csv(f"{LAB}/sample_r3.csv")[["patch_id", "zona", "estrato"]]
    l = pd.read_csv(f"{LAB}/labels_claude_r3.csv")[["patch_id", "label"]]
    sc = pd.read_csv("outputs/patch_eval/scores_all_v3.csv")
    d = s.merge(l, on="patch_id").merge(sc, on="patch_id")
    print("== r3: etiquetas por estrato ==")
    t = pd.crosstab(d.estrato, d.label).reindex(["alto", "medio", "bajo"])
    t["n"] = t.sum(axis=1)
    t["si/(si+no+otro)"] = (t.get("si", 0) / (t[[c for c in ("si", "no", "otro") if c in t]].sum(axis=1))).round(3)
    print(t.to_string())
    print("\n== r3: etiquetas por zona ==")
    print(pd.crosstab(d.zona, d.label).to_string())

    # referencia conjunta r1+r2 (aleatoria/estratificada por etiqueta débil) + r3 (sesgada)
    old = pd.read_csv("outputs/patch_eval/scores_v3.csv")[["patch_id", "label", "top100", "frac05", "max"]]
    allp = pd.concat([old.assign(fuente="r1r2"), d[["patch_id", "label", "top100", "frac05", "max"]].assign(fuente="r3")])
    allp = allp[allp.label != "dudoso"]
    print(f"\n== conjunto r1+r2+r3: n={len(allp)}, positivos={(allp.label == 'si').sum()} (sesgado hacia candidatos del modelo) ==")
    for nom, x in (("todo(no+otro)", allp), ("sin_otro", allp[allp.label.isin(["si", "no"])])):
        y = (x.label == "si").astype(int)
        if y.nunique() < 2:
            continue
        for sc_ in ("top100", "frac05", "max"):
            print(f"{nom:14s} {sc_:7s} AUC={roc_auc_score(y, x[sc_]):.3f} AP={average_precision_score(y, x[sc_]):.3f} (base {y.mean():.3f})")
    print("\n'si' de r3:", d[d.label == "si"][["patch_id", "estrato", "top100", "frac05"]].to_string(index=False))


if __name__ == "__main__":
    main()
