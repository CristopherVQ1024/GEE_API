"""Prueba real: detecta tala selectiva en una zona cualquiera a partir de coordenadas.

Descarga Sentinel-2 L2A (STAC público, sin Earth Engine), arma las 15 bandas del par de años
(temporada seca jun–sep, igual que el dataset), predice con el ensamble de folds y guarda en
outputs/predictions/<name>/:
  prob.tif            probabilidad por píxel (0–1), GeoTIFF en UTM
  features.tif        las 15 bandas de entrada (para repetir la predicción o usar la API)
  detecciones.geojson / detecciones.kml   polígonos detectados (abrir el KML en Google Earth)
  panel.png           RGB antes, RGB después, caída de NDFI y detecciones
  resumen.json        resumen, calidad por cuadro y lista de detecciones

Confianza por detección según su probabilidad media: alta (>= 0,75), media (0,6–0,75), baja (< 0,6);
las de confianza baja se guardan pero no se suman al resumen.
Calidad por cuadro de 256 px según el ruido de dNDFI en bosque (neblina/nubes residuales):
  ok (< 0,041), baja (0,041–0,08), no_evaluable (> 0,08); umbrales = p90 y criterio de exclusión del dataset.

    python -m src.predict_area --lon -69.95 --lat -12.90 --size_km 5 --years 2024 2025 --name prueba1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from pyproj import Transformer
from scipy import ndimage as ndi

from src.evaluate import get_device, load_config, load_fold_checkpoints
from src.refine_labels import refine
from src.s2_local import FEATURE_BANDS, features_from_composites, load_season

PATCH = 256
NOISE_LOW, NOISE_BAD = 0.041, 0.08
MIN_PX = 5
CONF_HIGH, CONF_MID = 0.75, 0.6
GROUP_PX = 3


def utm_epsg(lon: float, lat: float) -> int:
    return (32700 if lat < 0 else 32600) + int((lon + 180) // 6) + 1


def predict_mosaic(models: list[dict], feats: np.ndarray, device: torch.device, tta: bool) -> np.ndarray:
    """Ventanas de 256 px con paso 128 y promedio de solapes; ensamble de folds."""
    _, h, w = feats.shape
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    ys = list(range(0, h - PATCH + 1, PATCH // 2))
    xs = list(range(0, w - PATCH + 1, PATCH // 2))
    for m in models:
        mean = np.array(m["norm_stats"]["mean"], np.float32).reshape(-1, 1, 1)
        std = np.clip(np.array(m["norm_stats"]["std"], np.float32).reshape(-1, 1, 1), 1e-6, None)
        x_all = torch.from_numpy(np.nan_to_num((feats - mean) / std, nan=0.0)).to(device)
        for y in ys:
            for x in xs:
                t = x_all[None, :, y:y + PATCH, x:x + PATCH]
                with torch.no_grad():
                    p = torch.sigmoid(m["model"](t))
                    if tta:
                        for d in ((2,), (3,), (2, 3)):
                            p = p + torch.sigmoid(m["model"](t.flip(d))).flip(d)
                        p = p / 4
                acc[y:y + PATCH, x:x + PATCH] += p[0, 0].cpu().numpy()
                cnt[y:y + PATCH, x:x + PATCH] += 1
        del x_all
    return acc / np.maximum(cnt, 1)


def tile_quality(feats: np.ndarray, prob: np.ndarray) -> list[dict]:
    rows = []
    for r in range(feats.shape[1] // PATCH):
        for c in range(feats.shape[2] // PATCH):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            a = feats[:, sl[0], sl[1]]
            _, info = refine(a, np.zeros(a.shape[1:], np.uint8))
            noise = info["noise"]
            q = "ok" if noise < NOISE_LOW else ("baja" if noise <= NOISE_BAD else "no_evaluable")
            p = np.sort(prob[sl].ravel())[::-1]
            rows.append({"fila": r, "col": c, "ruido_dNDFI": round(noise, 4), "calidad": q,
                         "valido": round(float(np.isfinite(a[9]).mean()), 3),
                         "puntaje_top100": round(float(p[:100].mean()), 3), "frac_detectada": round(float((p > 0.5).mean()), 4)})
    return rows


def detections(prob: np.ndarray, thr: float, transform: Affine, crs: str, quality: np.ndarray):
    from rasterio.features import shapes
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union

    import shapely
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    det = prob > thr
    # fragmentos a menos de ~30 m forman un mismo evento, ej. un camino con sus patios
    lab, n = ndi.label(ndi.binary_dilation(det, iterations=GROUP_PX), structure=np.ones((3, 3)))
    lab[~det] = 0
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        npx = int(comp.sum())
        if npx < MIN_PX:
            continue
        sub_t = transform * Affine.translation(sl[1].start, sl[0].start)
        geom = [shape(g) for g, v in shapes(comp.astype(np.uint8), mask=comp, transform=sub_t) if v == 1]
        poly = unary_union(geom)
        cy, cx = ndi.center_of_mass(comp)
        q = quality[sl[0].start + int(cy), sl[1].start + int(cx)]
        pm = float(prob[sl][comp].mean())
        conf = "alta" if pm >= CONF_HIGH else ("media" if pm >= CONF_MID else "baja")
        ll = shapely.transform(poly, lambda c: np.column_stack(tr.transform(c[:, 0], c[:, 1])))
        out.append({"id": len(out) + 1, "area_ha": round(npx * 100 / 1e4, 2), "prob_media": round(pm, 3), "confianza": conf,
                    "lon": round(ll.representative_point().x, 6), "lat": round(ll.representative_point().y, 6), "calidad": q, "geometry": mapping(ll)})
    out.sort(key=lambda d: -d["prob_media"] * d["area_ha"])
    for k, d in enumerate(out, start=1):
        d["id"] = k
    return out


def write_kml(dets: list[dict], path: Path, name: str) -> None:
    def ring(coords):
        return " ".join(f"{x:.6f},{y:.6f},0" for x, y in coords)
    marks = []
    for d in dets:
        polys = [d["geometry"]["coordinates"]] if d["geometry"]["type"] == "Polygon" else d["geometry"]["coordinates"]
        geo = "".join(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring(p[0])}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
                      for p in polys)
        marks.append(f"<Placemark><name>{d['id']} {d['confianza']}</name><description>{d['area_ha']} ha, prob {d['prob_media']}, calidad {d['calidad']}"
                     f"</description><styleUrl>#det</styleUrl><MultiGeometry><Point><coordinates>{d['lon']},{d['lat']},0</coordinates></Point>"
                     f"{geo}</MultiGeometry></Placemark>")
    path.write_text('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                    f"<name>{name}</name><Style id=\"det\"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>"
                    "<PolyStyle><color>400000ff</color></PolyStyle></Style>" + "".join(marks) + "</Document></kml>", encoding="utf-8")


def panel(c0, c1, feats, prob, thr, quality_rows, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rgb(c):
        x = np.nan_to_num(np.stack([c[2], c[1], c[0]], -1))
        return np.clip((x / 0.12) ** 0.8, 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    ax[0].imshow(rgb(c0)); ax[0].set_title("Antes (RGB)")
    ax[1].imshow(rgb(c1)); ax[1].set_title("Después (RGB)")
    ax[2].imshow(feats[FEATURE_BANDS.index("dNDFI")], cmap="RdYlGn", vmin=-0.4, vmax=0.4); ax[2].set_title("Caída de NDFI (rojo)")
    ax[3].imshow(rgb(c1)); ax[3].contour(prob > thr, levels=[0.5], colors="red", linewidths=0.8)
    ax[3].set_xlim(-0.5, prob.shape[1] - 0.5); ax[3].set_ylim(prob.shape[0] - 0.5, -0.5)
    ax[3].set_title("Detecciones (rojo)")
    for q in quality_rows:
        if q["calidad"] != "ok":
            color = "yellow" if q["calidad"] == "baja" else "magenta"
            for a in ax:
                a.add_patch(plt.Rectangle((q["col"] * PATCH, q["fila"] * PATCH), PATCH, PATCH, fill=False, ec=color, lw=1.5, ls="--"))
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title + "   (cuadros punteados: amarillo = calidad baja, magenta = no evaluable por neblina)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--size_km", type=float, default=5.0, help="lado del cuadrado, máx. 10 km")
    ap.add_argument("--years", type=int, nargs=2, default=[2024, 2025], help="año antes y año después")
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", default=None, help="por defecto paths.checkpoints de config.yaml")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    if a.size_km > 10:
        raise SystemExit("size_km máximo 10 (memoria acotada)")

    cfg, dev = load_config(), get_device()
    if dev.type != "cuda":
        raise SystemExit("Sin GPU disponible: usar .venv-gpu")
    epsg = utm_epsg(a.lon, a.lat)
    crs = f"EPSG:{epsg}"
    cx, cy = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(a.lon, a.lat)
    n = max(1, math.ceil(a.size_km * 100 / PATCH))
    size_px = n * PATCH
    x0 = round((cx - size_px * 5) / 10) * 10
    y1 = round((cy + size_px * 5) / 10) * 10
    transform = Affine(10, 0, x0, 0, -10, y1)
    out = Path(cfg["paths"]["outputs"]) / "predictions" / a.name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Zona {size_px * 10 / 1000:.2f} km × {size_px * 10 / 1000:.2f} km en {crs}; descargando Sentinel-2 {a.years[0]} y {a.years[1]}...", flush=True)
    c0, n0 = load_season(x0, y1, crs, a.years[0], size_px, size_px)
    c1, n1 = load_season(x0, y1, crs, a.years[1], size_px, size_px)
    if n0 == 0 or n1 == 0:
        raise SystemExit(f"Sin escenas Sentinel-2 (antes {n0}, después {n1})")
    feats, valid = features_from_composites(c0, c1)
    print(f"Escenas: {n0} + {n1}; píxeles válidos {valid.mean():.0%}", flush=True)

    models = load_fold_checkpoints(a.ckpt or cfg["paths"]["checkpoints"], cfg["model"], dev)
    prob = predict_mosaic(models, feats, dev, a.tta)
    prob[~valid] = 0

    qrows = tile_quality(feats, prob)
    qmap = np.empty(prob.shape, dtype=object)
    for q in qrows:
        qmap[q["fila"] * PATCH:(q["fila"] + 1) * PATCH, q["col"] * PATCH:(q["col"] + 1) * PATCH] = q["calidad"]
    dets = detections(prob, a.thr, transform, crs, qmap)

    prof = dict(driver="GTiff", width=size_px, height=size_px, crs=crs, transform=transform, compress="deflate")
    with rasterio.open(out / "prob.tif", "w", count=1, dtype="float32", **prof) as dst:
        dst.write(prob[None])
    with rasterio.open(out / "features.tif", "w", count=15, dtype="float32", nodata=np.nan, **prof) as dst:
        dst.write(feats)
        dst.descriptions = tuple(FEATURE_BANDS)
    (out / "detecciones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": d["geometry"], "properties": {k: v for k, v in d.items() if k != "geometry"}} for d in dets]}))
    write_kml(dets, out / "detecciones.kml", a.name)
    ok_dets = [d for d in dets if d["calidad"] != "no_evaluable" and d["confianza"] != "baja"]
    summary = {"nombre": a.name, "centro": [a.lon, a.lat], "lado_km": size_px * 10 / 1000, "anios": a.years,
               "escenas": [n0, n1], "validos": round(float(valid.mean()), 3), "modelo": a.ckpt or cfg["paths"]["checkpoints"],
               "umbral": a.thr, "tta": a.tta,
               "n_detecciones": len(ok_dets), "n_baja_confianza": sum(d["confianza"] == "baja" for d in dets), "area_detectada_ha": round(sum(d["area_ha"] for d in ok_dets), 2),
               "cuadros_no_evaluables": sum(q["calidad"] == "no_evaluable" for q in qrows),
               "cuadros": qrows, "detecciones": [{k: v for k, v in d.items() if k != "geometry"} for d in dets]}
    (out / "resumen.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    panel(c0, c1, feats, prob, a.thr, qrows, out / "panel.png",
          f"{a.name}: {len(ok_dets)} detecciones de confianza alta o media, {summary['area_detectada_ha']} ha ({a.years[0]}→{a.years[1]})")

    print(f"\n{len(ok_dets)} detecciones de confianza alta o media, {summary['area_detectada_ha']} ha "
          f"({summary['n_baja_confianza']} de confianza baja aparte); "
          f"cuadros no evaluables por neblina: {summary['cuadros_no_evaluables']}/{len(qrows)}")
    for d in ok_dets[:15]:
        print(f"  #{d['id']:>3}  {d['area_ha']:>6} ha  prob {d['prob_media']:.2f} ({d['confianza']})  "
              f"{d['lat']:.5f}, {d['lon']:.5f}  calidad {d['calidad']}")
    print(f"\nSalida en {out}/ (abrir detecciones.kml en Google Earth y panel.png)")


if __name__ == "__main__":
    main()
