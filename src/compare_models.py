"""Comparación pareada de modelos a nivel de parche: Madre de Dios (fotointerpretación) y DETER test2026.

Lee los puntajes ya guardados por evaluate_patch_level y evaluate_deter. El primer tag es la referencia;
para cada otro modelo reporta el AUC y el bootstrap pareado de la diferencia (IC 95 % y P(>= referencia)).
En Madre de Dios también da el AUC con rangos por zona, porque Tambopata se puntúa con el modelo de su fold
y Tahuamanu con el ensamble y sus escalas no son comparables.

    python -m src.compare_models v3 v4 v5 v6
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as A

tags = sys.argv[1:]
ref = tags[0]
rng = np.random.default_rng(0)
B = 2000


def paired(y, sa, sb):
    y, sa, sb = map(np.asarray, (y, sa, sb))
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(A(y[i], sb[i]) - A(y[i], sa[i]))
    d = np.array(d)
    return f"dAUC={np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}; {np.percentile(d, 97.5):+.3f}] P(>=ref)={(d >= 0).mean():.2f}"


m = {t: pd.read_csv(f"outputs/patch_eval/scores_{t}.csv").set_index("patch_id") for t in tags}
base = m[ref]
base = base[base.label.isin(["si", "no", "otro"])]
y = (base.label == "si").astype(int)
print("== Madre de Dios, top100 (102 parches, 11 si) ==")
for t in tags:
    s = m[t].loc[base.index, "top100"]
    zr = s.groupby(base.zona).rank(pct=True)
    zones = {z: round(A(y[base.zona == z], s[base.zona == z]), 3) for z in ("tambopata", "tahuamanu")}
    line = f"{t}: pooled {A(y, s):.3f} | rank-por-zona {A(y, zr):.3f} | {zones}"
    if t != ref:
        line += " | vs " + ref + " " + paired(y, base["top100"], s)
    print(line)

print("\n== DETER test2026, top100 ==")
dd = {t: pd.read_csv(f"outputs/deter_eval/test2026/patches_{t}.csv").set_index("patch_id") for t in tags}
x = dd[ref]
x = x[(x.cs_frac >= 0.02) | x.clean]
yd = (x.cs_frac >= 0.02).astype(int)
for t in tags:
    s = dd[t].loc[x.index, "top100"]
    line = f"{t}: AUC {A(yd, s):.3f}"
    if t != ref:
        line += " | vs " + ref + " " + paired(yd, x["top100"], s)
    print(line)
