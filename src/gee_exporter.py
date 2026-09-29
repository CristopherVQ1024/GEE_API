"""
gee_exporter.py
================
Módulo de extracción y preprocesamiento en Google Earth Engine (GEE).

Genera, para cada ventana temporal definida en config.yaml, un stack
multibanda de Sentinel-2 (reflectancia + índices espectrales + textura)
y calcula diferencias multitemporales (dNDVI, dNBR) entre dos periodos,
que son la señal principal para detectar disturbios sutiles como la tala
selectiva (a diferencia de la deforestación total, que Hansen GFC ya
captura bien).

Uso típico (ver notebooks/01_exploratory_analysis.ipynb):

    import ee
    from src.gee_exporter import (
        init_ee, load_config, get_roi_geometry,
        get_sentinel2_composite, build_feature_stack,
        compute_change_layers, export_grid_to_drive,
    )

    cfg = load_config()
    init_ee(cfg)
    roi = get_roi_geometry(cfg)
    ...

Requiere: earthengine-api, geemap, pyyaml
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import ee
import yaml

CONFIG_PATH_DEFAULT = Path(__file__).resolve().parents[1] / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Configuración / autenticación
# ---------------------------------------------------------------------------

def load_config(path: Optional[str] = None) -> dict:
    """Carga config.yaml como diccionario."""
    cfg_path = Path(path) if path else CONFIG_PATH_DEFAULT
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_ee(cfg: dict) -> None:
    """
    Autentica e inicializa Earth Engine.

    IMPORTANTE: desde 2025 GEE exige un Google Cloud Project REGISTRADO
    (nivel no comercial) para inicializar. Si nunca lo hiciste:
      1. Ve a https://code.earthengine.google.com/register
      2. Crea/registra un proyecto (elige "Unpaid usage" -> académico/investigación)
      3. Copia el Project ID a config.yaml -> gee.project_id
    En Colab, ee.Authenticate() abre un flujo OAuth en el navegador.
    """
    project_id = cfg["gee"]["project_id"]
    if project_id == "TU-PROYECTO-GEE-AQUI":
        raise ValueError(
            "Configura gee.project_id en config/config.yaml con tu Project ID "
            "de Google Cloud registrado para Earth Engine antes de continuar."
        )
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)


# ---------------------------------------------------------------------------
# Área de estudio
# ---------------------------------------------------------------------------

def get_roi_geometry(cfg: dict) -> ee.Geometry:
    """Construye la geometría del ROI a partir del bounding box en config.yaml."""
    lon_min, lat_min, lon_max, lat_max = cfg["roi"]["bbox"]
    return ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], proj="EPSG:4326", geodesic=False)


# ---------------------------------------------------------------------------
# Sentinel-2: máscara de nubes + composite
# ---------------------------------------------------------------------------

def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Enmascara nubes y cirros usando la banda QA60 de Sentinel-2 L2A."""
    qa = image.select("QA60")
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )
    return image.updateMask(mask).divide(10000).copyProperties(image, image.propertyNames())


def get_sentinel2_composite(cfg: dict, roi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """
    Filtra Sentinel-2 L2A por ROI/fechas/nubosidad, enmascara nubes y
    devuelve la mediana temporal (composite robusto a outliers/nubes
    residuales) recortada al ROI.
    """
    s2_cfg = cfg["sentinel2"]
    coll = (
        ee.ImageCollection(s2_cfg["collection"])
        .filterBounds(roi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", s2_cfg["max_cloud_pct"]))
        .map(_mask_s2_clouds)
    )
    composite = coll.median().clip(roi)
    return composite


# ---------------------------------------------------------------------------
# Índices espectrales y textura
# ---------------------------------------------------------------------------

def compute_spectral_indices(composite: ee.Image) -> ee.Image:
    """NDVI y NBR (Normalized Burn Ratio, sensible a pérdida de canopy/humedad)."""
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = composite.normalizedDifference(["B8", "B12"]).rename("NBR")
    return ndvi.addBands(nbr)


def compute_nir_texture(cfg: dict, composite: ee.Image) -> ee.Image:
    """
    Textura (desviación estándar) del NIR en una ventana kxk.
    La tala selectiva rompe la homogeneidad del dosel: aumenta la
    heterogeneidad espacial del NIR incluso cuando el NDVI medio apenas cambia.
    """
    radius_px = cfg["sentinel2"]["texture_kernel_radius_px"]
    kernel = ee.Kernel.square(radius=radius_px, units="pixels")
    nir = composite.select("B8")
    sigma_nir = nir.reduceNeighborhood(reducer=ee.Reducer.stdDev(), kernel=kernel).rename("sigma_NIR")
    return sigma_nir


def build_feature_stack(cfg: dict, roi: ee.Geometry, start_date: str, end_date: str) -> ee.Image:
    """
    Construye el tensor multibanda final para una ventana temporal:
    [B2, B3, B4, B8, B11, B12, NDVI, sigma_NIR] -> 8 canales,
    coherente con model.in_channels en config.yaml.
    """
    composite = get_sentinel2_composite(cfg, roi, start_date, end_date)
    bands = cfg["sentinel2"]["bands"]
    indices = compute_spectral_indices(composite)
    sigma_nir = compute_nir_texture(cfg, composite)

    stack = (
        composite.select(bands)
        .addBands(indices.select("NDVI"))
        .addBands(sigma_nir)
    )
    return stack.toFloat()


# ---------------------------------------------------------------------------
# Diferencias multitemporales (señal principal de disturbio)
# ---------------------------------------------------------------------------

def compute_change_layers(cfg: dict, roi: ee.Geometry, before_window_id: str, after_window_id: str) -> ee.Image:
    """
    Calcula dNDVI y dNBR entre dos ventanas temporales definidas en
    config.yaml -> temporal_windows. Un dNDVI/dNBR negativo y localizado
    (no un cambio estacional difuso en toda la escena) es la firma
    espectral clásica de degradación/tala selectiva.
    """
    windows = {w["id"]: w for w in cfg["temporal_windows"]}
    if before_window_id not in windows or after_window_id not in windows:
        raise KeyError("Ventana temporal no encontrada en config.yaml -> temporal_windows")

    w_before, w_after = windows[before_window_id], windows[after_window_id]

    comp_before = get_sentinel2_composite(cfg, roi, w_before["start_date"], w_before["end_date"])
    comp_after = get_sentinel2_composite(cfg, roi, w_after["start_date"], w_after["end_date"])

    ndvi_before = comp_before.normalizedDifference(["B8", "B4"])
    ndvi_after = comp_after.normalizedDifference(["B8", "B4"])
    nbr_before = comp_before.normalizedDifference(["B8", "B12"])
    nbr_after = comp_after.normalizedDifference(["B8", "B12"])

    d_ndvi = ndvi_after.subtract(ndvi_before).rename("dNDVI")
    d_nbr = nbr_after.subtract(nbr_before).rename("dNBR")

    return d_ndvi.addBands(d_nbr).toFloat().clip(roi)


# ---------------------------------------------------------------------------
# Exportación (tiling)
# ---------------------------------------------------------------------------

def _lonlat_deg_per_pixel(scale_m: float, lat_deg: float) -> tuple[float, float]:
    """Aproxima el tamaño en grados de un píxel de `scale_m` metros en `lat_deg`."""
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat_deg))
    return scale_m / meters_per_deg_lon, scale_m / meters_per_deg_lat


def build_export_grid(cfg: dict, roi: ee.Geometry) -> list[dict]:
    """
    Genera una grilla de tiles (parches) de patch_size_px x patch_size_px
    (a la resolución `scale_m`) que cubren el ROI. Devuelve una lista de
    dicts {id, lon_min, lat_min, lon_max, lat_max} para exportar cada uno
    por separado (necesario porque Export.image.toDrive tiene límites de
    tamaño por tarea).
    """
    lon_min, lat_min, lon_max, lat_max = cfg["roi"]["bbox"]
    patch_px = cfg["tiling"]["patch_size_px"]
    scale_m = cfg["tiling"]["scale_m"]
    lat_center = (lat_min + lat_max) / 2
    deg_lon, deg_lat = _lonlat_deg_per_pixel(scale_m, lat_center)
    tile_w_deg = deg_lon * patch_px
    tile_h_deg = deg_lat * patch_px

    tiles = []
    lat = lat_min
    row = 0
    while lat < lat_max:
        lon = lon_min
        col = 0
        while lon < lon_max:
            tiles.append({
                "id": f"tile_r{row:03d}_c{col:03d}",
                "lon_min": lon, "lat_min": lat,
                "lon_max": min(lon + tile_w_deg, lon_max),
                "lat_max": min(lat + tile_h_deg, lat_max),
            })
            lon += tile_w_deg
            col += 1
        lat += tile_h_deg
        row += 1
    return tiles


def filter_tiles_by_forest_relevance(tiles: list[dict], min_forest_frac: float = 0.05,
                                      max_forest_frac: float = 0.98) -> "ee.FeatureCollection":
    """
    (Opcional, recomendado para no gastar cuota de export en tiles
    irrelevantes) Filtra tiles usando Hansen treecover2000: descarta tiles
    casi sin bosque (agua/urbano) y tiles de bosque totalmente intacto sin
    ningún borde (baja probabilidad de tala selectiva activa), quedándote
    con la franja de "interfaz" donde ocurre la mayor parte del disturbio.
    Devuelve una ee.FeatureCollection lista para .getInfo() o exportar.
    """
    gfc = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    treecover = gfc.select("treecover2000").divide(100)

    feats = []
    for t in tiles:
        geom = ee.Geometry.Rectangle([t["lon_min"], t["lat_min"], t["lon_max"], t["lat_max"]])
        frac = treecover.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e9
        ).get("treecover2000")
        feats.append(ee.Feature(geom, {"id": t["id"], "forest_frac": frac}))

    fc = ee.FeatureCollection(feats)
    return fc.filter(
        ee.Filter.And(
            ee.Filter.gt("forest_frac", min_forest_frac),
            ee.Filter.lt("forest_frac", max_forest_frac),
        )
    )


def export_tile_to_drive(image: ee.Image, tile: dict, description: str, folder: str,
                          scale_m: int = 10, max_pixels: float = 1e13) -> ee.batch.Task:
    """Lanza (start) una tarea de exportación de un tile a Google Drive en GeoTIFF."""
    geom = ee.Geometry.Rectangle([tile["lon_min"], tile["lat_min"], tile["lon_max"], tile["lat_max"]])
    task = ee.batch.Export.image.toDrive(
        image=image.clip(geom),
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=geom,
        scale=scale_m,
        crs="EPSG:4326",
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()
    return task


def export_grid_to_drive(cfg: dict, image: ee.Image, tiles: list[dict], name_prefix: str,
                          folder: str = "madre_de_dios_logging") -> list[ee.batch.Task]:
    """
    Exporta `image` (ya recortada/preparada) tile por tile a Google Drive.
    NOTA: cada llamada a Export cuenta como una tarea en la cola de GEE;
    para ROIs grandes esto puede ser docenas de tareas. Revisa el progreso
    en https://code.earthengine.google.com/tasks
    """
    scale_m = cfg["tiling"]["scale_m"]
    max_pixels = cfg["tiling"]["max_pixels_export"]
    tasks = []
    for t in tiles:
        desc = f"{name_prefix}_{t['id']}"
        tasks.append(export_tile_to_drive(image, t, desc, folder, scale_m, max_pixels))
    return tasks


if __name__ == "__main__":
    cfg = load_config()
    init_ee(cfg)
    roi = get_roi_geometry(cfg)
    print("ROI cargado:", cfg["roi"]["name"], cfg["roi"]["bbox"])

    pair = cfg["default_change_pair"]
    stack_after = build_feature_stack(
        cfg, roi,
        cfg["temporal_windows"][-1]["start_date"],
        cfg["temporal_windows"][-1]["end_date"],
    )
    change = compute_change_layers(cfg, roi, pair["before"], pair["after"])
    full_image = stack_after.addBands(change)
    print("Bandas del stack final:", full_image.bandNames().getInfo())

    tiles = build_export_grid(cfg, roi)
    print(f"Grilla generada: {len(tiles)} tiles de {cfg['tiling']['patch_size_px']}px")
