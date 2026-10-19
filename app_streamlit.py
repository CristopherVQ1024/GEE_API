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

import json
import re
import subprocess
import sys
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
import yaml
from rasterio.warp import calculate_default_transform, reproject, Resampling
from streamlit_folium import st_folium

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "config.yaml"
ROOT = Path(__file__).resolve().parent
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"
LOGS_DIR = ROOT / "outputs" / "logs"


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


def mapa_existente() -> None:
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


def job_activo() -> bool:
    return subprocess.run(["pgrep", "-f", r"python[0-9.]* -m src\.predict_area"], capture_output=True).returncode == 0


def lanzar_job(lon: float, lat: float, size_km: float, years: tuple[int, int], name: str, tta: bool) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=10G", "-p", "MemorySwapMax=2G",
        "env", "OMP_NUM_THREADS=4",
        sys.executable, "-m", "src.predict_area",
        "--lon", str(lon), "--lat", str(lat), "--size_km", str(size_km),
        "--years", str(years[0]), str(years[1]), "--name", name,
    ]
    if tta:
        cmd.append("--tta")
    log = open(LOGS_DIR / f"predict_{name}.log", "w", encoding="utf-8")
    subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    st.session_state["job_name"] = name


def mostrar_resultado(name: str) -> None:
    out = PREDICTIONS_DIR / name
    resumen = json.loads((out / "resumen.json").read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Detecciones", resumen["n_detecciones"])
    c2.metric("Área detectada (ha)", resumen["area_detectada_ha"])
    c3.metric("Baja confianza", resumen["n_baja_confianza"])
    no_eval = resumen.get("cuadros_no_evaluables", 0)
    if no_eval:
        st.warning(f"{no_eval} cuadros no evaluables por neblina o nubes residuales.")
    st.image(str(out / "panel.png"), width="stretch")

    lon, lat = resumen["centro"]
    m = folium.Map(location=[lat, lon], zoom_start=14, tiles="Esri.WorldImagery")
    gj = out / "detecciones.geojson"
    if gj.exists():
        folium.GeoJson(
            json.loads(gj.read_text(encoding="utf-8")),
            style_function=lambda _: {"color": "red", "weight": 2, "fillOpacity": 0.3},
        ).add_to(m)
    st_folium(m, width=900, height=500, key=f"res_{name}")

    kml = out / "detecciones.kml"
    if kml.exists():
        st.download_button("Descargar KML para Google Earth", kml.read_bytes(), file_name=f"{name}.kml")


@st.fragment(run_every=3)
def estado_job() -> None:
    name = st.session_state.get("job_name")
    if not name:
        return
    out = PREDICTIONS_DIR / name
    if (out / "resumen.json").exists():
        st.success(f"Listo: {name}")
        return
    if job_activo():
        st.info(f"Procesando {name}. Descarga de Sentinel-2 y predicción, unos minutos.")
        log = LOGS_DIR / f"predict_{name}.log"
        if log.exists():
            st.code("\n".join(log.read_text(encoding="utf-8", errors="ignore").splitlines()[-4:]))
    else:
        st.error(f"El proceso terminó sin resultado. Revisa outputs/logs/predict_{name}.log")


def analizar_zona() -> None:
    st.session_state.setdefault("lon", -69.35)
    st.session_state.setdefault("lat", -12.75)

    col_map, col_form = st.columns([3, 2])
    with col_map:
        st.caption("Haz clic en el mapa para elegir el centro de la zona.")
        m = folium.Map(location=[st.session_state["lat"], st.session_state["lon"]], zoom_start=9, tiles="Esri.WorldImagery")
        folium.Marker([st.session_state["lat"], st.session_state["lon"]]).add_to(m)
        click = st_folium(m, width=700, height=450, key="selector").get("last_clicked")
        if click and (round(click["lng"], 4), round(click["lat"], 4)) != (round(st.session_state["lon"], 4), round(st.session_state["lat"], 4)):
            st.session_state["lon"], st.session_state["lat"] = round(click["lng"], 5), round(click["lat"], 5)
            st.rerun()

    with col_form:
        lon = st.number_input("Longitud", key="lon", format="%.5f")
        lat = st.number_input("Latitud", key="lat", format="%.5f")
        size_km = st.slider("Lado de la zona (km)", 1.0, 10.0, 5.0, 0.5)
        y0, y1 = st.columns(2)
        antes = y0.number_input("Año antes", 2017, 2030, 2024)
        despues = y1.number_input("Año después", 2017, 2030, 2025)
        tta = st.checkbox("TTA, más lento y más estable")
        default_name = re.sub(r"[^a-z0-9_]", "", f"zona_{lon:.2f}_{lat:.2f}".replace("-", "m").replace(".", "p"))
        name = re.sub(r"[^A-Za-z0-9_]", "_", st.text_input("Nombre de la salida", default_name)) or default_name

        busy = job_activo()
        if st.button("Analizar zona", type="primary", disabled=busy or despues <= antes):
            lanzar_job(lon, lat, size_km, (int(antes), int(despues)), name, tta)
            st.rerun()
        if busy and not st.session_state.get("job_name"):
            st.warning("Ya hay un análisis corriendo.")
        if despues <= antes:
            st.caption("El año después debe ser mayor que el año antes.")

    estado_job()
    name_done = st.session_state.get("job_name")
    if name_done and (PREDICTIONS_DIR / name_done / "resumen.json").exists():
        mostrar_resultado(name_done)


def resultados_previos() -> None:
    carpetas = sorted(p.parent.name for p in PREDICTIONS_DIR.glob("*/resumen.json")) if PREDICTIONS_DIR.exists() else []
    if not carpetas:
        st.info("Aún no hay análisis en outputs/predictions.")
        return
    mostrar_resultado(st.selectbox("Análisis guardado", carpetas))


def main() -> None:
    st.set_page_config(page_title="Detección de Tala Selectiva — Madre de Dios", layout="wide")
    st.title("Detección de Tala Selectiva — Madre de Dios")
    st.caption("Sentinel-2 y U-Net. Elige una zona, el modelo compara dos años y marca la tala nueva.")
    t1, t2, t3 = st.tabs(["Analizar zona", "Análisis guardados", "Mapa de máscaras"])
    with t1:
        analizar_zona()
    with t2:
        resultados_previos()
    with t3:
        mapa_existente()


if __name__ == "__main__":
    main()
