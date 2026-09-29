# Detección de tala selectiva en Madre de Dios mediante Deep Learning y análisis multitemporal de Sentinel-2 en la nube

Implementación del artículo de investigación (Ingeniería de Sistemas). Este
repo cubre desde la extracción de datos en Google Earth Engine hasta el
entrenamiento del modelo, la validación y un prototipo funcional
(API + mapa interactivo).

## ⚠️ Léeme primero: caveat sobre el ground truth (afecta tu Metodología)

Al revisar `geobosques.minam.gob.pe` (sept. 2026) se confirmó que:

- **Sí existen y son descargables/consultables**: WMS raster de "Bosque y
  pérdida de bosque" (deforestación TOTAL, 2001-2025), WMS de "Alerta
  temprana" (actualización ~21-26 días), y una API REST de **estadísticas
  agregadas** (stock/pérdida por departamento/provincia/distrito — no a
  nivel de píxel/polígono).
- **No se confirmó** un endpoint público de descarga masiva de **polígonos**
  del submódulo "Degradación" (tala selectiva), aunque el submódulo existe
  conceptualmente en la plataforma.

Esto invalida la idea original de "usar polígonos históricos de GeoBosques
como etiquetas de tala selectiva" tal cual. La solución implementada aquí
(ver `src/labels.py`) es un **esquema de etiquetas en dos niveles**, que es
metodológicamente defendible y común en la literatura cuando no hay ground
truth oficial de degradación:

1. **Pseudo-etiquetas heurísticas** (todo el ROI, para entrenar en volumen):
   dentro de bosque (Hansen `treecover2000`), que NO sea deforestación total
   (Hansen `lossyear`), y con caída localizada de dNDVI/dNBR entre dos
   periodos Sentinel-2.
2. **Etiquetas validadas por fotointerpretación** (subconjunto pequeño,
   ~30-60 parches): tú/tu asesor digitalizan a mano los polígonos donde el
   patrón de tala selectiva es inequívoco. **Este subconjunto es tu test
   set real** para las métricas del artículo — nunca se usa para entrenar.

Adicionalmente:
- **Gestiona en paralelo** una solicitud formal a PNCBMCC (contacto en
  `geobosques.minam.gob.pe/geobosque/view/contacto.php`) pidiendo el dataset
  vectorial de Degradación. Si llega a tiempo, reemplaza/complementa el
  nivel 1 con datos oficiales reales — el pipeline no cambia, solo la fuente.
- **FODEX (Iñapari)**: no confirmaste acceso a las parcelas de campo. El
  pipeline **no depende** de esa fuente; si consigues acceso, `labels.py` es
  el lugar para integrarla (regresión ΔAGB vs. probabilidad del modelo).

GeoBosques "Alerta Temprana" y GFW (GLAD/GLAD-S2 alerts) se usan como
**validación cruzada independiente** — no como etiqueta de entrenamiento — y
habilitan el análisis opcional de "latencia de detección" (Fase 5).

## Estructura del repositorio

```
madre_de_dios_logging_detection/
├── config/
│   └── config.yaml              # ROI, ventanas temporales, hiperparámetros, rutas
├── src/
│   ├── gee_exporter.py          # Extracción/preprocesamiento Sentinel-2 en GEE
│   ├── labels.py                # Estrategia de ground truth (ver caveat arriba)
│   ├── spatial_split.py         # Spatial K-Fold (bloques 20x20 km, anti-leakage)
│   ├── dataset.py               # PyTorch Dataset
│   ├── models/unet.py           # U-Net (segmentation-models-pytorch)
│   ├── utils/losses.py          # BCE + Dice Loss
│   ├── utils/metrics.py         # IoU, F1, Precision, Recall
│   ├── train.py                 # Entrenamiento con validación cruzada espacial
│   ├── evaluate.py              # Validación holdout + validación empírica externa
│   └── api.py                   # Backend FastAPI (sirve el modelo)
├── app_streamlit.py             # Frontend: mapa interactivo (Folium)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb   # Listo para Google Colab
├── data/{raw,processed}/        # (vacío hasta que exportes tus datos)
├── outputs/{checkpoints,metrics,predictions}/
└── requirements.txt
```

## Cómo correr el proyecto, paso a paso

### Fase 0 — Setup

1. **Google Earth Engine**: desde 2025 GEE exige un **Google Cloud Project
   registrado** (nivel no comercial) — ya no basta con una cuenta personal
   antigua.
   - Regístralo en https://code.earthengine.google.com/register
   - Copia el Project ID a `config/config.yaml` → `gee.project_id`
   - La primera vez, `init_ee()` (en `gee_exporter.py`) te pedirá
     autenticarte vía navegador (`ee.Authenticate()`).
2. Instala dependencias: `pip install -r requirements.txt` (en Colab, usa
   el notebook que ya trae la celda de instalación).
3. Revisa/ajusta el ROI en `config.yaml` → `roi.bbox`. Viene precargado con
   el corredor de la Carretera Interoceánica Sur entre **Iberia e Iñapari**
   (provincia de Tahuamanu, Madre de Dios) — zona representativa de
   fragmentación/tala en la literatura para la frontera agrícola amazónica.
   **Confírmalo visualmente** en el notebook antes de exportar todo.

### Fase 1 — Extracción (`src/gee_exporter.py`)

Genera, por cada ventana temporal en `config.yaml`, un stack de 8 bandas
(B2, B3, B4, B8, B11, B12, NDVI, sigma_NIR) y las diferencias
multitemporales (dNDVI, dNBR) entre el par `default_change_pair`. Exporta
a Google Drive en tiles de 256×256 px vía `export_grid_to_drive`.

### Fase 2 — Etiquetado (`src/labels.py`)

Construye la pseudo-etiqueta heurística (`build_training_label_image`) y
provee utilidades para:
- Rasterizar polígonos externos (`rasterize_reference_polygons`) — úsalo en
  cuanto tengas el subconjunto de fotointerpretación o datos de GeoBosques.
- Consultar alertas GFW (`query_gfw_glad_alerts`) para validación cruzada.

### Fase 3 — Split espacial (`src/spatial_split.py`)

Con el CSV de metadatos de tus parches (`patch_id, lon_center, lat_center`),
corre `spatial_kfold_split` para asignar bloques de 20×20 km a folds,
reservando un bloque completo como `external_holdout` (nunca visto en
entrenamiento). Guarda el resultado como
`data/processed/patches_metadata.csv`.

### Fase 4 — Entrenamiento (`src/train.py`)

```bash
python -m src.train --metadata data/processed/patches_metadata.csv
```

Entrena un modelo por cada fold numérico (excluyendo `external_holdout`),
con U-Net + encoder ResNet34 preentrenado, pérdida BCE+Dice (por el fuerte
desbalance de clases: la tala ocupa &lt;5% del área), AdamW +
ReduceLROnPlateau, y early stopping. Guarda métricas por fold en
`outputs/metrics/cv_results_summary.json` — **esto va directo a tu sección
"Resultados del modelo"**.

### Fase 5 — Validación (`src/evaluate.py`)

```bash
# Validación en el bloque geográfico reservado (ensemble de todos los folds)
python -m src.evaluate --metadata data/processed/patches_metadata.csv

# + validación empírica externa contra un dataset oficial ya rasterizado
python -m src.evaluate --metadata ... \
    --prediction_mosaic outputs/predictions/mosaico.tif \
    --reference_raster data/processed/geobosques_alineado.tif
```

La segunda corrida es tu **aporte diferenciador**: concordancia píxel a
píxel entre tu modelo y una fuente oficial/validada, en una zona/periodo
que el modelo nunca vio. Opcional y de alto valor para "novedad": compara
la fecha en que tu modelo detecta un cambio vs. la fecha de la alerta GLAD
equivalente (`labels.query_gfw_glad_alerts`) — si detectas antes, es un
resultado fuerte y fácil de graficar como línea de tiempo.

### Fase 6 — Prototipo

```bash
# Backend
uvicorn src.api:app --reload --port 8000

# Frontend
streamlit run app_streamlit.py
```

## Mapeo metodología ↔ secciones del artículo

| Fase | Script | Sección del paper |
|---|---|---|
| 0-1. Setup + extracción | `gee_exporter.py` | Metodología |
| 2. Etiquetado | `labels.py` | Metodología (justifica el esquema de 2 niveles) |
| 3. Split espacial | `spatial_split.py` | Metodología (anti-leakage) |
| 4. Entrenamiento | `train.py` | Metodología + Resultados (métricas por fold) |
| 5. Validación externa | `evaluate.py` | Resultados / Discusión (tu diferenciador) |
| 6. Prototipo | `api.py`, `app_streamlit.py` | Anexo / demo de la sustentación |

## Próximos pasos recomendados

1. Corre el notebook exploratorio, confirma visualmente el ROI y calibra
   los umbrales de dNDVI/dNBR con tus propios histogramas.
2. Digitaliza el subconjunto de fotointerpretación (30-60 parches) — es lo
   más importante para que tus métricas sean creíbles ante un revisor.
3. Envía la solicitud a PNCBMCC por el dataset de Degradación en paralelo;
   no bloquees el proyecto esperando la respuesta.
4. Exporta un lote pequeño de prueba (2-3 tiles) antes del export completo
   para validar formato/bandas.
