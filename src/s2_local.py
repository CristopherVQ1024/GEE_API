"""Réplica local (sin Earth Engine) de las 15 bandas del dataset v2.

Reproduce `feature_stack` del notebook 01_exploratory_analysis_v2 (Colab) leyendo Sentinel-2 L2A
desde el catálogo STAC público de Element84 (AWS, colección `sentinel-2-c1-l2a`, ya armonizada:
reflectancia = DN·1e-4 − 0,1, igual que COPERNICUS/S2_SR_HARMONIZED/10000).

Diferencias conocidas con GEE:
  - máscara de nubes con SCL (Sen2Cor) en vez de Cloud Score+ (cs_cdf ≥ 0,6);
  - desmezcla con mínimos cuadrados completamente restringidos (suma 1, no negativos) resueltos
    por enumeración de subconjuntos, equivalente a `ee.Image.unmix(E, True, True)`.
`python -m src.s2_local --check tambopata_r010_c010` compara contra un parche exportado por GEE.

Uso como librería:
    feats, valid = patch_features(x0, y1, crs, year_before, year_after)   # (15, 256, 257)
"""
from __future__ import annotations

import argparse
import itertools
import warnings
from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi

warnings.filterwarnings("ignore", category=UserWarning, module="pystac")

STAC = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"
ASSETS = ["blue", "green", "red", "nir", "swir16", "swir22"]  # B2 B3 B4 B8 B11 B12
SCL_OK = (4, 5, 6, 11)  # vegetación, suelo, agua, nieve (resto: nube, sombra, cirros, saturado, sin dato)
SEASON = ("06-01", "09-30")
FEATURE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI_t1", "NDFI_t0", "NDFI_t1", "dNDFI",
                 "NPV_t1", "Soil_t1", "sigma_NIR_t0", "sigma_NIR_t1", "dNBR"]
# Endmembers del notebook v2 [B2, B3, B4, B8, B11, B12]
ENDMEMBERS = {
    "GV": [0.05, 0.09, 0.04, 0.61, 0.30, 0.10],
    "Shade": [0, 0, 0, 0, 0, 0],
    "NPV": [0.14, 0.17, 0.22, 0.30, 0.55, 0.30],
    "Soil": [0.20, 0.30, 0.34, 0.58, 0.60, 0.58],
    "Cloud": [0.90, 0.96, 0.80, 0.78, 0.72, 0.65],
}
EM_NAMES = list(ENDMEMBERS)
E = np.array([ENDMEMBERS[k] for k in EM_NAMES], dtype=np.float64).T  # (6 bandas, 5 endmembers)


POOL = 32  # lecturas HTTP en paralelo por worker; el cuello es la latencia de AWS, no la RAM (no suma memoria)
HTTP_TIMEOUT = 60  # s; sin esto una lectura colgada bloquea el bloque entero


@lru_cache(maxsize=1)
def _device():
    """Mediana y desmezcla van en GPU (CUDA). En CPU solo si no hay GPU, y avisando."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("AVISO s2_local: CUDA no disponible, calculando en CPU", flush=True)
    torch.set_num_threads(2)
    return torch.device("cpu")


@lru_cache(maxsize=1)
def _configure_rio():
    import odc.stac
    odc.stac.configure_rio(cloud_defaults=True, aws={"aws_unsigned": True},
                           GDAL_HTTP_TIMEOUT=HTTP_TIMEOUT, GDAL_HTTP_MAX_RETRY=5, GDAL_HTTP_RETRY_DELAY=3)
    return True


@lru_cache(maxsize=1)
def _client():
    from pystac_client import Client
    return Client.open(STAC, timeout=HTTP_TIMEOUT)


def _nanmedian_t(x):
    """Mediana sobre el eje 0 ignorando NaN, igual que np.nanmedian (promedia los dos centrales si n es par)."""
    import torch
    n = (~torch.isnan(x)).sum(0, keepdim=True)
    s = torch.sort(x, dim=0).values  # torch deja los NaN al final
    lo = torch.gather(s, 0, ((n - 1).clamp(min=0) // 2))
    hi = torch.gather(s, 0, (n // 2).clamp(max=x.shape[0] - 1))
    med = (lo + hi)[0] / 2
    return torch.where(n[0] > 0, med, torch.full_like(med, float("nan")))


def load_season(x0: float, y1: float, crs: str, year: int, width: int = 257, height: int = 256,
                max_cloud: float = 80.0, season=SEASON):
    """Mediana (6, H, W) de la temporada seca, enmascarando nubes con SCL. NaN = sin dato."""
    import odc.stac
    from odc.geo.geobox import GeoBox
    from affine import Affine
    from pyproj import Transformer

    gb = GeoBox((height, width), Affine(10, 0, x0, 0, -10, y1), crs)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xs = [x0, x0 + 10 * width]; ys = [y1 - 10 * height, y1]
    lons, lats = to_ll.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    items = _client().search(collections=[COLLECTION], bbox=bbox,
                             datetime=f"{year}-{season[0]}/{year}-{season[1]}",
                             query={"eo:cloud_cover": {"lt": max_cloud}}).item_collection()
    if len(items) == 0:
        return np.full((6, height, width), np.nan, np.float32), 0
    _configure_rio()
    ds = odc.stac.load(items, bands=ASSETS + ["scl"], geobox=gb, groupby="solar_day",
                       resampling="nearest", pool=POOL, fail_on_error=False)
    # El cubo queda en uint16 en RAM (t×H×W×2 B por banda); cada banda se pasa sola a la GPU y allí se
    # convierte, enmascara y reduce. Antes: 6 bandas en float32 + np.stack + copia de nanmedian (~5–8 GB/bloque).
    import torch
    dev = _device()
    n_t = len(ds.time)
    ok = torch.from_numpy(np.isin(ds["scl"].values, SCL_OK)).to(dev)
    med = np.empty((len(ASSETS), height, width), np.float32)
    for i, a in enumerate(ASSETS):
        v = torch.from_numpy(ds[a].values.view(np.int16)).to(dev).to(torch.int32) & 0xFFFF
        v = v.to(torch.float32)
        v = torch.where((v > 0) & ok, v * 1e-4 - 0.1, torch.full_like(v, float("nan")))
        med[i] = _nanmedian_t(v).cpu().numpy()
        del v
    del ds, ok
    return med, n_t


def _subsets():
    k = E.shape[1]
    return [s for r in range(1, k + 1) for s in itertools.combinations(range(k), r)]


def unmix_fcls(refl: np.ndarray) -> np.ndarray:
    """Fracciones (5, H, W) con suma 1 y no negativas: mejor subconjunto factible por mínimos cuadrados."""
    import torch
    dev = _device()
    b, h, w = refl.shape
    X = refl.reshape(b, -1).astype(np.float64)
    good = np.all(np.isfinite(X), 0)
    Xg = torch.from_numpy(X[:, good]).to(dev)
    best_err = torch.full((Xg.shape[1],), float("inf"), dtype=torch.float64, device=dev)
    best = torch.zeros((E.shape[1], Xg.shape[1]), dtype=torch.float64, device=dev)
    for s in _subsets():
        Es = E[:, s]
        # mínimos cuadrados con sum(f)=1 vía multiplicador de Lagrange (sistema KKT); K es chico (≤6×6):
        # se resuelve en numpy y solo el producto por los píxeles va a la GPU
        k = len(s)
        K = np.zeros((k + 1, k + 1)); K[:k, :k] = Es.T @ Es; K[:k, k] = 1; K[k, :k] = 1
        try:
            Kinv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(K)
        Es_t = torch.from_numpy(Es).to(dev)
        A = torch.from_numpy(Kinv[:k, :k] @ Es.T).to(dev)  # (k, 6)
        c = torch.from_numpy(Kinv[:k, k]).to(dev)[:, None]  # término del multiplicador (fila de unos)
        sol = A @ Xg + c
        feas = (sol >= -1e-9).all(0)
        err = ((Xg - Es_t @ sol) ** 2).sum(0)
        upd = feas & (err < best_err)
        best_err = torch.where(upd, err, best_err)
        full = torch.zeros_like(best); full[list(s)] = sol
        best = torch.where(upd[None], full, best)
    out = np.full((E.shape[1], X.shape[1]), np.nan)
    out[:, good] = best.cpu().numpy()
    return out.reshape(E.shape[1], h, w).astype(np.float32)


def ndfi_parts(refl):
    fr = dict(zip(EM_NAMES, unmix_fcls(refl)))
    gvs = fr["GV"] / np.maximum(1 - fr["Shade"], 1e-6)
    ndfi = (gvs - (fr["NPV"] + fr["Soil"])) / np.maximum(gvs + fr["NPV"] + fr["Soil"], 1e-6)
    return ndfi, fr["NPV"], fr["Soil"]


def sigma3(x):
    """Desviación estándar 3×3 (poblacional, como ee.Reducer.stdDev)."""
    m = ndi.uniform_filter(np.nan_to_num(x), 3, mode="nearest")
    m2 = ndi.uniform_filter(np.nan_to_num(x) ** 2, 3, mode="nearest")
    s = np.sqrt(np.clip(m2 - m * m, 0, None))
    return np.where(np.isfinite(x), s, np.nan)


def nd(a, b):
    return (a - b) / np.where(np.abs(a + b) < 1e-9, np.nan, a + b)


def features_from_composites(c0, c1):
    n0, _, _ = ndfi_parts(c0)
    n1, npv1, soil1 = ndfi_parts(c1)
    B = dict(zip(["B2", "B3", "B4", "B8", "B11", "B12"], c1))
    nbr0 = nd(c0[3], c0[5]); nbr1 = nd(c1[3], c1[5])
    bands = {**B, "NDVI_t1": nd(c1[3], c1[2]), "NDFI_t0": n0, "NDFI_t1": n1, "dNDFI": n1 - n0,
             "NPV_t1": npv1, "Soil_t1": soil1, "sigma_NIR_t0": sigma3(c0[3]), "sigma_NIR_t1": sigma3(c1[3]),
             "dNBR": nbr1 - nbr0}
    feats = np.stack([bands[k] for k in FEATURE_BANDS]).astype(np.float32)
    valid = np.isfinite(c0[3]) & np.isfinite(c1[3])
    feats[:, ~valid] = np.nan
    return feats, valid


def patch_features(x0, y1, crs, year_before, year_after, width=257, height=256):
    c0, n0 = load_season(x0, y1, crs, year_before, width, height)
    c1, n1 = load_season(x0, y1, crs, year_after, width, height)
    feats, valid = features_from_composites(c0, c1)
    return feats, valid, (n0, n1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="patch_id exportado por GEE para comparar")
    a = ap.parse_args()
    import rasterio
    with rasterio.open(f"data/processed/images/{a.check}.tif") as s:
        ref = s.read().astype(np.float32); t = s.transform; crs = s.crs.to_string()
    feats, valid, n = patch_features(t.c, t.f, crs, 2024, 2025, ref.shape[2], ref.shape[1])
    print(f"escenas usadas: {n}  válidos local {valid.mean():.3f}")
    for i, b in enumerate(FEATURE_BANDS):
        r, l = ref[i], feats[i]
        m = np.isfinite(r) & np.isfinite(l)
        cc = np.corrcoef(r[m], l[m])[0, 1]
        print(f"{b:13s} r={cc:6.3f}  media GEE {np.mean(r[m]):8.4f}  local {np.mean(l[m]):8.4f}  "
              f"MAE {np.mean(np.abs(r[m] - l[m])):.4f}")


if __name__ == "__main__":
    main()
