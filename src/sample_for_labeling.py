"""Selecciona parches estratificados para fotointerpretar (test real).

Estratos: zona (Tambopata / Tahuamanu) x (con positivos / sin positivos).
Tahuamanu se sobre-representa en la parte "con positivos" porque casi no tiene.

    python -m src.sample_for_labeling --n 80 --out data/labeling/sample.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="data/processed/patches_metadata.csv")
    ap.add_argument("--out", default="data/labeling/sample.csv")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.metadata)
    df["zona"] = df["patch_id"].str.split("_").str[0]
    df["con_pos"] = df["pos_frac"] > 0.001
    per = a.n // 4
    parts = []
    for (_, _), g in df.groupby(["zona", "con_pos"]):
        parts.append(g.sample(min(per, len(g)), random_state=a.seed))
    s = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s[["patch_id", "zona", "con_pos", "pos_frac", "fold"]].to_csv(a.out, index=False)
    print(s.groupby(["zona", "con_pos"]).size())
    print(f"-> {a.out} ({len(s)} parches)")


if __name__ == "__main__":
    main()
