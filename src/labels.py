"""
labels.py
=========
Estrategia de "ground truth" para tala selectiva / degradación forestal.

CAVEAT IMPORTANTE (léelo antes de programar nada más — va directo a tu
sección de Metodología del artículo):

    Al revisar geobosques.minam.gob.pe (sept. 2026) se confirmó que el
    submódulo "Degradación" existe conceptualmente (mencionado en
    /geobosque/view/acerca.php) pero NO se encontró un endpoint público de
    descarga masiva de POLÍGONOS de tala selectiva. Lo que sí está
    confirmado y es descargable/consultable:
      - WMS raster de "Bosque y pérdida de bosque" 2001-2025 (deforestación
        TOTAL, no degradación parcial).
      - WMS raster de "Alerta temprana" (actualización ~21-26 días).
      - API REST de estadísticas agregadas (stock/pérdida) a nivel
        nacional/región/provincia/distrito, NO a nivel de píxel/polígono.
    Esto significa que "usar polígonos históricos de GeoBosques como
    etiquetas" (tu plan original) no es directamente viable con lo público
    hoy. Dos caminos, no excluyentes:

    (A) CONTACTAR a PNCBMCC (geobosques.minam.gob.pe/geobosque/view/contacto.php)
        para solicitar formalmente el dataset vectorial del submódulo de
        Degradación. Como estudiante/universidad, es razonable pedirlo con
        fines de investigación — pero no lo asumas disponible a tiempo.

    (B) (Recomendado para no bloquear el proyecto) Construir un ESQUEMA DE
        ETIQUETAS EN DOS NIVELES, que es metodológicamente defendible y de
        hecho común en la literatura de teledetección cuando no hay ground
        truth oficial de degradación (ver p.ej. los papers de disturbio con
        BFAST/CNN que mencionaste):
          1. Pseudo-etiquetas heurísticas (débiles, en TODO el ROI):
             disturbio = está dentro de bosque (Hansen treecover2000 alto)
             Y NO es deforestación total (Hansen lossyear) Y presenta una
             caída localizada de dNDVI/dNBR + aumento de sigma_NIR entre
             dos periodos. Sirven para pre-entrenar/entrenar en volumen.
          2. Etiquetas validadas por fotointerpretación (fuertes, en un
             subconjunto pequeño, ~30-60 parches de 256x256): tú (o tu
             asesor) revisan visualmente el compuesto RGB + dNDVI/dNBR en
             geemap/QGIS y digitalizan a mano los polígonos donde SÍ se ve
             el patrón espacial de tala selectiva (trochas, claros de
             dosel, patrón de espina de pescado). Este subconjunto es tu
             verdadero "test set" reservado para las métricas de tu
             artículo (IoU/F1/Precision/Recall) — no lo uses para entrenar.

    GeoBosques "Alerta Temprana" y GFW "GLAD/GLAD-S2 alerts" siguen siendo
    útiles como VALIDACIÓN CRUZADA INDEPENDIENTE (no como etiqueta de
    entrenamiento pixel-perfect), comparando la ubicación/fecha de tus
    detecciones contra sus alertas oficiales — de hecho esto es tu
    diferenciador de "velocidad de detección" (Fase 5 del README).

Este módulo implementa el camino (B): construcción de la máscara heurística
y utilidades para rasterizar/incorporar polígonos manuales o de terceros
(GeoBosques si consigues acceso, GFW, digitalización propia) cuando existan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import ee
import numpy as np


# ---------------------------------------------------------------------------
# 1. Máscaras base desde Hansen Global Forest Change
# ---------------------------------------------------------------------------

def get_hansen_image(asset_id: str = "UMD/hansen/global_forest_change_2025_v1_13") -> ee.Image:
    return ee.Image(asset_id)


def get_forest_baseline_mask(hansen: ee.Image, treecover_threshold: int = 30) -> ee.Image:
    """
    Bosque base (año 2000) según Hansen: treecover2000 >= umbral (%).
    Se usa para restringir el análisis a área forestal y no perder tiempo
    de cómputo/entrenamiento en agua, urbano o pastizales ya establecidos.
    """
    return hansen.select("treecover2000").gte(treecover_threshold).rename("forest_mask")


def get_total_deforestation_mask(hansen: ee.Image, year_start: int, year_end: int) -> ee.Image:
    """
    Máscara de deforestación TOTAL (stand-replacement) de Hansen entre
    year_start y year_end (ambos en 2001-2025). lossyear está codificado
    como año - 2000 (p.ej. 24 = 2024).
    Esta clase se EXCLUYE de "tala selectiva": si ya no queda bosque, no es
    degradación parcial, es deforestación total (otro problema, ya bien
    cubierto por GFW/Hansen).
    """
    loss = hansen.select("loss")
    lossyear = hansen.select("lossyear")
    y0, y1 = year_start - 2000, year_end - 2000
    in_range = lossyear.gte(y0).And(lossyear.lte(y1))
    return loss.And(in_range).rename("total_deforestation_mask")


# ---------------------------------------------------------------------------
# 2. Pseudo-etiqueta heurística de degradación (nivel píxel)
# ---------------------------------------------------------------------------

def compute_disturbance_proxy_mask(
    change_image: ee.Image,
    forest_mask: ee.Image,
    total_deforestation_mask: ee.Image,
    ndvi_drop_threshold: float = -0.10,
    nbr_drop_threshold: float = -0.08,
) -> ee.Image:
    """
    Construye la pseudo-etiqueta binaria de "posible tala selectiva /
    degradación" combinando:
      - estar dentro del bosque base (forest_mask == 1)
      - NO ser deforestación total en el mismo periodo
      - mostrar una caída de dNDVI y/o dNBR mayor al umbral (valores
        negativos porque dNDVI = NDVI_after - NDVI_before)

    Los umbrales por defecto son un punto de partida conservador tomado de
    literatura de detección de disturbio con Landsat/Sentinel; AJÚSTALOS
    inspeccionando histogramas de dNDVI/dNBR de tu ROI en el notebook
    exploratorio (busca el "codo" que separe ruido estacional de disturbio
    real) antes de generar el dataset completo.

    Devuelve una imagen de una banda 'proxy_disturbance' (0/1).
    """
    ndvi_drop = change_image.select("dNDVI").lte(ndvi_drop_threshold)
    nbr_drop = change_image.select("dNBR").lte(nbr_drop_threshold)
    disturbance_signal = ndvi_drop.Or(nbr_drop)

    proxy = (
        forest_mask.eq(1)
        .And(total_deforestation_mask.Not())
        .And(disturbance_signal)
    )
    return proxy.rename("proxy_disturbance").toByte()


def build_training_label_image(
    hansen_asset_id: str,
    change_image: ee.Image,
    treecover_threshold: int = 30,
    deforestation_year_start: int = 2022,
    deforestation_year_end: int = 2025,
    ndvi_drop_threshold: float = -0.10,
    nbr_drop_threshold: float = -0.08,
) -> ee.Image:
    """Función de conveniencia: encadena las funciones anteriores en un solo paso."""
    hansen = get_hansen_image(hansen_asset_id)
    forest_mask = get_forest_baseline_mask(hansen, treecover_threshold)
    total_loss_mask = get_total_deforestation_mask(hansen, deforestation_year_start, deforestation_year_end)
    return compute_disturbance_proxy_mask(
        change_image, forest_mask, total_loss_mask, ndvi_drop_threshold, nbr_drop_threshold
    )


# ---------------------------------------------------------------------------
# 3. Ingesta de polígonos externos (GeoBosques si obtienes acceso, GFW,
#    o tu propia digitalización manual de fotointerpretación)
# ---------------------------------------------------------------------------

def rasterize_reference_polygons(
    vector_path: str,
    reference_raster_path: str,
    out_raster_path: str,
    burn_value: int = 1,
    all_touched: bool = False,
) -> str:
    """
    Rasteriza un shapefile/GeoJSON de polígonos (GeoBosques, GFW, o
    digitalización manual propia) a la misma grilla/resolución/CRS que un
    raster de referencia (p.ej. uno de tus tiles de Sentinel-2 ya
    exportados), para poder compararlo píxel a píxel contra las
    predicciones del modelo o usarlo como máscara de entrenamiento fuerte.

    Requiere: geopandas, rasterio
    """
    import geopandas as gpd
    import rasterio
    from rasterio import features

    gdf = gpd.read_file(vector_path)
    with rasterio.open(reference_raster_path) as ref:
        gdf = gdf.to_crs(ref.crs)
        out_shape = (ref.height, ref.width)
        transform = ref.transform
        meta = ref.meta.copy()

    shapes = ((geom, burn_value) for geom in gdf.geometry if geom is not None)
    mask = features.rasterize(
        shapes=shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )

    meta.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_raster_path, "w", **meta) as dst:
        dst.write(mask, 1)

    return out_raster_path


def query_gfw_glad_alerts(geometry_geojson: dict, start_date: str, end_date: str,
                           api_key: Optional[str] = None) -> dict:
    """
    Consulta la API pública de Global Forest Watch (data-api.globalforestwatch.org)
    para obtener alertas GLAD/GLAD-S2 dentro de una geometría y rango de
    fechas — útil como validación cruzada independiente y para el análisis
    de latencia de detección (¿tu modelo detecta el disturbio antes que el
    ciclo de alertas oficial?).

    Documentación de la API: https://data-api.globalforestwatch.org/
    Necesitarás registrarte para una API key gratuita en
    https://www.globalforestwatch.org/ (Developer account) para consultas
    en volumen; consultas puntuales de prueba pueden no requerirla.

    Esta función es un STUB con la estructura de la petición: revisa la
    documentación oficial vigente para el endpoint exacto del dataset que
    elijas (p.ej. 'gfw_integrated_alerts' o 'umd_glad_sentinel2_alerts'),
    ya que la API de GFW cambia de versión con cierta frecuencia.
    """
    import requests

    base_url = "https://data-api.globalforestwatch.org/dataset/umd_glad_sentinel2_alerts/latest/query"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    sql = (
        "SELECT latitude, longitude, alert_date, confidence "
        f"FROM data WHERE alert__date BETWEEN '{start_date}' AND '{end_date}'"
    )
    payload = {"sql": sql, "geometry": geometry_geojson}

    resp = requests.post(base_url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# 4. Utilidades de exploración de umbrales (para el notebook)
# ---------------------------------------------------------------------------

def suggest_change_thresholds_from_sample(dndvi_values: np.ndarray, dnbr_values: np.ndarray,
                                           percentile: float = 5.0) -> dict:
    """
    Sugiere umbrales de dNDVI/dNBR a partir de una muestra de valores
    (extraídos con reduceRegion + sampleRectangle o exportando un tile de
    prueba y leyéndolo con rasterio). Usa el percentil bajo (por defecto 5%)
    como punto de partida razonable: asume que la mayoría de los píxeles no
    cambia (ruido estacional/BRDF) y que la cola negativa concentra el
    disturbio real. AJUSTA visualmente comparando con el compuesto RGB.
    """
    return {
        "ndvi_drop_threshold": float(np.percentile(dndvi_values, percentile)),
        "nbr_drop_threshold": float(np.percentile(dnbr_values, percentile)),
    }
