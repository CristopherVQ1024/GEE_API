"""Parches Sentinel-2 (15 bandas) sobre polígonos oficiales de tala selectiva de INPE DETER.

DETER (INPE, Brasil) publica polígonos de alerta con clase y fecha; las clases CS_DESORDENADO
(tala selectiva desordenada/ilegal) y CS_GEOMETRICO (manejo forestal con patrón regular) son
tala selectiva mapeada por intérpretes. Es la referencia publicada que usamos para:
  - un benchmark de prueba en un año que el modelo nunca vio (par 2025→2026);
  - etiquetas reales para afinar el modelo (pares 2023→2024 y 2024→2025, en bloques alejados).

Para cada bloque de 10,24 km (1024×1024 px a 10 m, UTM local) con tala selectiva DETER en la ventana:
  1. compone la temporada seca (jun–sep) de los dos años con src/s2_local (réplica del notebook GEE);
  2. rasteriza DETER: banda 1 = tala selectiva en la ventana, banda 2 = tala selectiva de otras fechas,
     banda 3 = otras clases DETER (deforestación, minería, quema, degradación) en el periodo del par;
  3. corta 16 parches 256×256 → <out>/images, <out>/masks y <out>/metadata.csv.

    python -m src.deter_dataset --name test2026 --pair 2025 2026 --window 2026-06-15 2026-09-30 --n_blocks 32
    python -m src.deter_dataset --name train2025 --pair 2024 2025 --window 2025-06-15 2025-09-30 \
        --n_blocks 40 --exclude data/deter/test2026/blocks.csv --exclude_km 20
"""
from __future__ import annotations

import argparse
import json
import math
import urllib.parse
import urllib.request
import multiprocessing as mp
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_PX = 1024
PATCH = 256
BLOCK_M = BLOCK_PX * 10
WFS = "https://terrabrasilis.dpi.inpe.br/geoserver/deter-amz/ows"
ROOT = Path("data/deter")


def wfs_get(cql: str) -> dict:
    q = {"service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeName": "deter-amz:deter_amz",
         "outputFormat": "application/json", "srsName": "EPSG:4326", "CQL_FILTER": cql}
    url = f"{WFS}?{urllib.parse.urlencode(q)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return json.load(r)
        except Exception as e:  # HTTPError lleva el socket adjunto y no se puede enviar entre procesos
            err = f"{type(e).__name__}: {e}"
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"WFS DETER no responde ({err})")


def deter_cs(cache=ROOT / "raw/deter_cs_2022plus.geojson"):
    import geopandas as gpd
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(wfs_get("classname LIKE 'CS%' AND view_date>='2022-01-01'")))
    return gpd.read_file(cache)


def utm_epsg(lon, lat):
    z = int(math.floor((lon + 180) / 6)) + 1
    return (32700 if lat < 0 else 32600) + z


def select_blocks(cs, window, n_blocks, per_uf_max, exclude=None, exclude_km=20.0):
    from pyproj import Transformer
    w = cs[(cs.view_date >= window[0]) & (cs.view_date <= window[1])].copy()
    c = w.geometry.representative_point()
    w["lon"], w["lat"] = c.x, c.y
    w["epsg"] = [utm_epsg(lo, la) for lo, la in zip(w.lon, w.lat)]
    rows = []
    for epsg, g in w.groupby("epsg"):
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tr.transform(g.lon.values, g.lat.values)
        g = g.assign(bx=np.floor(np.asarray(x) / BLOCK_M).astype(int), by=np.floor(np.asarray(y) / BLOCK_M).astype(int))
        for (bx, by), gg in g.groupby(["bx", "by"]):
            back = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
            lo, la = back.transform((bx + .5) * BLOCK_M, (by + .5) * BLOCK_M)
            rows.append({"block_id": f"{epsg}_{bx}_{by}", "epsg": epsg, "bx": bx, "by": by, "lon": lo, "lat": la,
                         "uf": gg.uf.mode().iloc[0], "cs_km2": gg.areamunkm.sum(), "n_cs": len(gg),
                         "geometrico_frac": float((gg.classname == "CS_GEOMETRICO").mean())})
    b = pd.DataFrame(rows).sort_values("cs_km2", ascending=False)
    if exclude is not None and len(exclude):
        from math import radians, cos
        def far(r):
            d = np.hypot((exclude.lon - r.lon) * 111 * cos(radians(r.lat)), (exclude.lat - r.lat) * 111)
            return d.min() > exclude_km
        b = b[b.apply(far, axis=1)]
    # bloques con algo de tala visible (>=0,3 km²) y reparto entre estados (round-robin por ranking)
    b = b[b.cs_km2 >= 0.3]
    b["rank_uf"] = b.groupby("uf").cumcount()
    b = b[b.rank_uf < per_uf_max].sort_values(["rank_uf", "cs_km2"], ascending=[True, False])
    return b.head(n_blocks).reset_index(drop=True)


def process_block(block: dict, pair, window, out: Path) -> list[dict]:
    import geopandas as gpd
    import rasterio
    from affine import Affine
    from pyproj import Transformer
    from rasterio.features import rasterize

    from src.s2_local import features_from_composites, load_season

    bid, epsg = block["block_id"], int(block["epsg"])
    done = out / "blocks_done" / f"{bid}.json"
    if done.exists():
        return json.loads(done.read_text())
    x0, y1 = block["bx"] * BLOCK_M, (block["by"] + 1) * BLOCK_M
    crs = f"EPSG:{epsg}"
    # DETER de todas las clases en el bloque (antes que Sentinel-2: si el WFS está caído, falla sin descargar)
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lons, lats = tr.transform([x0, x0 + BLOCK_M, x0, x0 + BLOCK_M], [y1 - BLOCK_M, y1 - BLOCK_M, y1, y1])
    bb = f"{min(lons):.5f},{min(lats):.5f},{max(lons):.5f},{max(lats):.5f}"
    fc = wfs_get(f"BBOX(geom,{bb},'EPSG:4326')")
    c0, n0 = load_season(x0, y1, crs, pair[0], BLOCK_PX, BLOCK_PX)
    c1, n1 = load_season(x0, y1, crs, pair[1], BLOCK_PX, BLOCK_PX)
    feats, valid = features_from_composites(c0, c1)

    transform = Affine(10, 0, x0, 0, -10, y1)
    masks = np.zeros((3, BLOCK_PX, BLOCK_PX), np.uint8)
    if fc["features"]:
        d = gpd.GeoDataFrame.from_features(fc["features"], crs="EPSG:4326").to_crs(crs)
        d = d[d.geometry.notna() & ~d.geometry.is_empty]
        is_cs = d.classname.str.startswith("CS")
        in_win = (d.view_date >= window[0]) & (d.view_date <= window[1])
        period = (d.view_date >= f"{pair[0]}-06-01") & (d.view_date <= window[1])
        sets = [d[is_cs & in_win], d[is_cs & ~in_win], d[~is_cs & period]]
        for i, s in enumerate(sets):
            if len(s):
                masks[i] = rasterize(((g, 1) for g in s.geometry), out_shape=(BLOCK_PX, BLOCK_PX),
                                     transform=transform, fill=0, dtype="uint8", all_touched=False)
    rows = []
    (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(parents=True, exist_ok=True)
    for r in range(BLOCK_PX // PATCH):
        for c in range(BLOCK_PX // PATCH):
            sl = (slice(r * PATCH, (r + 1) * PATCH), slice(c * PATCH, (c + 1) * PATCH))
            pid = f"{out.name}_{bid}_r{r}_c{c}"
            px0, py1 = x0 + c * PATCH * 10, y1 - r * PATCH * 10
            prof = dict(driver="GTiff", width=PATCH, height=PATCH, crs=crs, transform=Affine(10, 0, px0, 0, -10, py1),
                        compress="deflate", tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(out / "images" / f"{pid}.tif", "w", count=15, dtype="float32", nodata=np.nan, **prof) as dst:
                dst.write(feats[:, sl[0], sl[1]])
            with rasterio.open(out / "masks" / f"{pid}.tif", "w", count=3, dtype="uint8", **prof) as dst:
                dst.write(masks[:, sl[0], sl[1]])
            lo, la = tr.transform(px0 + PATCH * 5, py1 - PATCH * 5)
            m = masks[:, sl[0], sl[1]]
            rows.append({"patch_id": pid, "block_id": bid, "uf": block["uf"], "epsg": epsg, "x0": px0, "y1": py1,
                         "lon_center": lo, "lat_center": la, "year_before": pair[0], "year_after": pair[1],
                         "cs_frac": float(m[0].mean()), "cs_other_frac": float(m[1].mean()),
                         "other_frac": float(m[2].mean()), "valid_frac": float(valid[sl].mean()),
                         "n_scenes_t0": n0, "n_scenes_t1": n1})
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(json.dumps(rows))
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    print(f"  {bid}: escenas {n0}+{n1}, RAM pico del worker {rss_gb:.1f} GB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--pair", nargs=2, type=int, required=True)
    ap.add_argument("--window", nargs=2, required=True, help="rango de view_date DETER (AAAA-MM-DD)")
    ap.add_argument("--n_blocks", type=int, default=32)
    ap.add_argument("--per_uf_max", type=int, default=10)
    ap.add_argument("--exclude", nargs="*", default=[], help="blocks.csv de otros conjuntos a evitar")
    ap.add_argument("--exclude_km", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()

    out = ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    cs = deter_cs()
    excl = pd.concat([pd.read_csv(f) for f in a.exclude]) if a.exclude else None
    blocks_csv = out / "blocks.csv"
    if blocks_csv.exists():
        blocks = pd.read_csv(blocks_csv)
    else:
        blocks = select_blocks(cs, a.window, a.n_blocks, a.per_uf_max, excl, a.exclude_km)
        blocks.to_csv(blocks_csv, index=False)
    print(blocks.groupby("uf").agg(n=("block_id", "size"), cs_km2=("cs_km2", "sum")).round(1).to_string(), flush=True)

    done = {p.stem for p in (out / "blocks_done").glob("*.json")}
    if len(done) < len(blocks):  # sin el WFS de DETER cada bloque fallaría tras minutos de reintentos
        try:
            wfs_get("classname = 'CS_DESORDENADO' AND view_date >= '2099-01-01'")
        except RuntimeError as e:
            raise SystemExit(f"Abortado: {e}. Relanzar cuando terrabrasilis responda (los bloques hechos se conservan).")

    rows, pending = [], blocks.to_dict("records")
    # spawn: cada worker arranca limpio (sin heredar memoria del padre ni un contexto CUDA)
    ctx = mp.get_context("spawn")
    for attempt in (1, 2):  # el segundo intento solo repite los bloques fallidos (timeouts de red)
        failed = []
        # max_tasks_per_child: el worker se recicla y devuelve su memoria (crece con bloques de muchas escenas)
        with ProcessPoolExecutor(a.workers, mp_context=ctx, max_tasks_per_child=3) as ex:
            futs = {ex.submit(process_block, b, a.pair, a.window, out): b for b in pending}
            for i, f in enumerate(as_completed(futs), 1):
                bid = futs[f]["block_id"]
                try:
                    rows += f.result()
                    print(f"[{i}/{len(futs)}] {bid} ok", flush=True)
                except Exception as e:  # un bloque fallido no detiene el resto
                    failed.append(futs[f])
                    print(f"[{i}/{len(futs)}] {bid} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
        if not failed:
            break
        pending = failed
        print(f"reintentando {len(failed)} bloque(s) fallido(s)", flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(out / "metadata.csv", index=False)
    print(f"{len(meta)} parches -> {out}")


if __name__ == "__main__":
    main()
