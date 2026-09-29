"""
app_streamlit.py
=================
Frontend simple del prototipo: mapa interactivo (Folium) que muestra el
área de estudio y, cuando existan, las máscaras de predicción exportadas
por src/evaluate.py::predict_and_export_mask.

Ejecutar (desde la raíz del repo):
    streamlit run app_streamlit.py

Este archivo es intencionalmente simple: su objetivo es dar un "prototipo
funcional" visual para tu sustentación, no ser una app de producción. Corre
sin necesidad de tener el modelo ya entrenado (muestra el ROI igualmente);
en cuanto generes máscaras de predicción con evaluate.py, aparecerán
automáticamente listadas para visualizar.
"""
from __future__ import annotations

from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
import yaml
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import st_folium

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"
PREDICTIONS_DIR = Path(__file__).resolve().parent / "outputs" / "predictions"


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reproject_mask_to_wgs84(path: str) -> tuple[np.ndarray, list]:
    """Reproyecta un GeoTIFF de máscara a EPSG:4326 y devuelve (array, bounds) para folium.raster_layers.ImageOverlay."""
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:4326", src.width, src.height, *src.bounds
        )
        dst_array = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        bounds = rasterio.transform.array_bounds(height, width, transform)  # (left, bottom, right, top)
        folium_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    return dst_array, folium_bounds


def main() -> None:
    st.set_page_config(page_title="Detección de Tala Selectiva — Madre de Dios", layout="wide")
    st.title("🌳 Detección de Tala Selectiva — Madre de Dios (Sentinel-2 + Deep Learning)")
    st.caption(
        "Prototipo funcional del artículo de investigación. Corredor Interoceánico Sur, "
        "provincia de Tahuamanu, Madre de Dios, Perú."
    )

    cfg = load_config()
    lon_min, lat_min, lon_max, lat_max = cfg["roi"]["bbox"]
    center_lat, center_lon = (lat_min + lat_max) / 2, (lon_min + lon_max) / 2

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("Capas disponibles")
        st.markdown(f"**ROI:** `{cfg['roi']['name']}`")
        st.markdown(f"**BBox:** `{cfg['roi']['bbox']}`")

        pred_files = sorted(PREDICTIONS_DIR.glob("*.tif")) if PREDICTIONS_DIR.exists() else []
        if pred_files:
            selected = st.selectbox(
                "Máscara de predicción a mostrar",
                options=[p.name for p in pred_files],
            )
        else:
            selected = None
            st.info(
                "Aún no hay máscaras en outputs/predictions/. "
                "Genera una con src/evaluate.py::predict_and_export_mask "
                "y aparecerá aquí automáticamente."
            )

        st.markdown("---")
        st.markdown(
            "**Leyenda**\n\n"
            "🟥 Rojo: píxeles clasificados como tala selectiva / degradación\n\n"
            "El polígono azul marca el área de estudio (ROI) configurada en `config.yaml`."
        )

    with col1:
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="blue", fill=False, weight=2, tooltip="Área de estudio (ROI)",
        ).add_to(m)

        if selected:
            mask_path = str(PREDICTIONS_DIR / selected)
            mask_array, bounds = reproject_mask_to_wgs84(mask_path)

            rgba = np.zeros((*mask_array.shape, 4), dtype=np.uint8)
            rgba[mask_array == 1] = [255, 0, 0, 160]  # rojo semitransparente

            folium.raster_layers.ImageOverlay(
                image=rgba, bounds=bounds, opacity=0.8, name="Predicción"
            ).add_to(m)
            folium.LayerControl().add_to(m)

        st_folium(m, width=900, height=650)


if __name__ == "__main__":
    main()
