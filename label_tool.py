"""Herramienta de fotointerpretación rápida (test real, nunca se usa para entrenar).

    streamlit run label_tool.py

Muestra por parche: RGB 2025 (B4,B3,B2), falso color SWIR (B12,B8,B4), dNDFI y las
etiquetas débiles. Se etiqueta a nivel de parche: ¿hay tala selectiva visible?
Las respuestas se guardan en data/labeling/labels.csv (se puede cerrar y retomar).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data/labeling/sample.csv"
LABELS = ROOT / "data/labeling/labels.csv"
IMG = ROOT / "data/processed/images"
MSK = ROOT / "data/processed/masks"
B = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI_t1", "NDFI_t0", "NDFI_t1", "dNDFI",
     "NPV_t1", "Soil_t1", "sigma_NIR_t0", "sigma_NIR_t1", "dNBR"]
OPCIONES = {"Sí – tala selectiva": "si", "No": "no", "Dudoso": "dudoso",
            "Otro (minería/agri/nube)": "otro"}


def stretch(x, lo=2, hi=98):
    a, b = np.nanpercentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def rgb(img, idx):
    return np.dstack([stretch(np.nan_to_num(img[B.index(i)])) for i in idx])


@st.cache_data
def load(pid):
    with rasterio.open(IMG / f"{pid}.tif") as s:
        img = s.read().astype("float32")
    with rasterio.open(MSK / f"{pid}.tif") as s:
        m = s.read(1)
    return img, m


def save(pid, val, nota):
    df = pd.read_csv(LABELS) if LABELS.exists() else pd.DataFrame(columns=["patch_id", "label", "nota"])
    df = df[df.patch_id != pid]
    df = pd.concat([df, pd.DataFrame([{"patch_id": pid, "label": val, "nota": nota}])])
    df.to_csv(LABELS, index=False)


st.set_page_config(layout="wide", page_title="Fotointerpretación")
if not SAMPLE.exists():
    st.error("Falta data/labeling/sample.csv: corre `python -m src.sample_for_labeling`.")
    st.stop()
sample = pd.read_csv(SAMPLE)
done = pd.read_csv(LABELS) if LABELS.exists() else pd.DataFrame(columns=["patch_id", "label", "nota"])
CL = ROOT / "data/labeling/labels_claude.csv"
solo_rev = False
if CL.exists():
    cl = pd.read_csv(CL).set_index("patch_id")["label"]
    solo_rev = st.sidebar.checkbox("Revisar solo lo que Claude marcó sí/dudoso", value=True)
    st.sidebar.caption("Se muestra la sugerencia de Claude para que la confirmes o corrijas.")
    if solo_rev:
        sample = sample[sample.patch_id.map(cl).isin(["si", "dudoso"])]
pend = sample[~sample.patch_id.isin(done.patch_id)]
st.progress(min(1.0, sample.patch_id.isin(done.patch_id).sum() / len(sample)), text=f"{sample.patch_id.isin(done.patch_id).sum()}/{len(sample)} etiquetados")
if pend.empty:
    st.success("¡Listo! Todos etiquetados.")
    st.stop()

row = pend.iloc[0]
pid = row.patch_id
img, m = load(pid)
st.subheader(f"{pid}  ·  {row.zona}")  # no mostramos pos_frac para no sesgar la decisión
if solo_rev:
    st.info(f"Sugerencia de Claude: **{cl[pid]}**")
fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
ax[0].imshow(rgb(img, ["B4", "B3", "B2"])); ax[0].set_title("RGB 2025")
ax[1].imshow(rgb(img, ["B12", "B8", "B4"])); ax[1].set_title("Falso color SWIR (B12,B8,B4)")
ax[2].imshow(img[B.index("dNDFI")], cmap="RdBu", vmin=-0.3, vmax=0.3); ax[2].set_title("dNDFI (rojo = pérdida)")
ax[3].imshow(rgb(img, ["B4", "B3", "B2"])); ax[3].contour(m > 0, levels=[0.5], colors="yellow", linewidths=1)
ax[3].set_title("RGB + etiqueta débil (amarillo)")
for a in ax: a.axis("off")
st.pyplot(fig); plt.close(fig)

nota = st.text_input("Nota (opcional)", key=f"n{pid}")
cols = st.columns(len(OPCIONES))
for c, (txt, val) in zip(cols, OPCIONES.items()):
    if c.button(txt, use_container_width=True, key=f"{pid}{val}"):
        save(pid, val, nota)
        st.rerun()
