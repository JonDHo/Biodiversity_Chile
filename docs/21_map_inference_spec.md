# Mapas multitemporales de facetas: especificación de inferencia

Cómo se producen los mapas anuales 2000–2026 de las siete facetas con el modelo final,
dónde corre cada parte, y qué decisiones se tomaron. Las decisiones son de J. Lopatin
(2026-09-02, sesión coordinada desde el Mac); la ejecución corre en el pod de Data Cube
Chile (`jupyter-jlopatin`).

**Addendum (2026-09-08): el modelo desplegado cambió.** Todo lo que sigue describe el
2D-CNN + MAE (`scripts/72_train_final_map_model.py --all-data`), que era el modelo final
cuando se escribió este documento. `docs/20` §9.7 muestra que sobre la topografía corregida
el MAE no gana ninguna faceta; el modelo desplegado pasó al 1D-CNN
(`scripts/77_train_final_map_model_c1d.py --all-data`, `C1D01_curve1d_kndvi`,
`results/models_unified_topofix/C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/`,
5 semillas × 28 épocas fijas). `src/biodiv/mapinfer.py` (`FacetEnsemble`) detecta la
arquitectura del checkpoint por su propio `input_shape`, así que el resto de este documento
-- decisiones D1-D13, tamaño del problema, dónde corre la inferencia -- sigue aplicando sin
cambios; sólo la fila "modelo" y "checkpoints" de §1 abajo describen el despliegue anterior.

## 1. Qué modelo y qué entrada

| pieza | valor | fuente |
|---|---|---|
| modelo | 1D-CNN `Pheno1D` (width B, sin trunk preentrenado), 14.535 parámetros. Superseded: 2D-CNN `PhenoNetS` inicializada desde MAE, 15.175 parámetros, ver addendum arriba | `docs/20` §9.7, `scripts/77` |
| checkpoints | 5 semillas, `results/models_unified_topofix/C1D01_curve1d_kndvi_raw100_pg-all_unified_ctr_FINAL_alldata/final/model_seed{0..4}.pt`, refit sobre las 3.102 parcelas, 28 épocas fijas | `logs/96` |
| entrada fenológica | serie **cruda** kNDVI del **píxel central**, ventana causal `y−2..y`, 100 pasos, media móvil de 5, curva 1D sin reshape (`FacetEnsemble.model_inputs`; era imagen `serpentine` 10×10 con el modelo superseded) | `biodiv.curves.interp_grid`, `biodiv.transforms1d` |
| contexto | 8 variables topográficas del píxel (`features.TOPO_VARS`) + bandera de terreno plano + `log10(área)` + 3 indicadores de estrato; 3 indicadores de faltante ⇒ 16 columnas | `ck["ctx_preprocessor"].columns_` |
| salidas | `lcbd_count_sorensen`, `pd_inext_q0/q1/q2`, `td_inext_q0/q1/q2` | `ck["targets"]` |

El código de inferencia es `src/biodiv/mapinfer.py` (núcleo numérico, probado sin
datacube en `tests/test_mapinfer.py`) y `scripts/73_map_inference.py` (driver por
tesela). `scripts/74_check_map_consistency.py` es la compuerta previa: sobre las 3.102
parcelas de entrenamiento, el camino de mapa debe reproducir **exactamente** el contexto,
la imagen y la salida del modelo del camino de entrenamiento, y el escalador de targets
debe cerrar el viaje de ida y vuelta. No se produce ningún mapa si esa compuerta falla.

## 2. Decisiones

| # | decisión | valor | por qué |
|---|---|---|---|
| D1 | extensión | envolvente de las parcelas unificadas (30,2–55,0°S) + 20 km, **solo vegetación nativa** según MapBiomas | el modelo se entrenó en parcelas de vegetación nativa; fuera de eso no hay soporte. Elección del autor (Q1) |
| D2 | máscara | clases nativas `{3, 59, 60, 61, 11, 12, 63, 66}` del mapa anual **más cercano** (`biodiv.mapbiomas.year_map`); 2025 y 2026 usan 2024 y llevan `map_delta` en los metadatos | MapBiomas termina en 2024 |
| D3 | área de parcela | **900 m²** (un píxel Landsat), constante | elección del autor (Q2): el mapa se lee como "diversidad esperada en una unidad de 900 m²". Está por encima del percentil 90 del pool (500 m²); es extrapolación leve y declarada |
| D4 | estrato | **`basal`** (protocolo de inventario, Living Trees) | 81 % del pool LCBD y 73 % del pool TD/PD; pendiente de confirmación explícita del autor |
| D5 | años | **2000–2026**; la ventana de 2000 (1998–2000) existe (L5 14 + 9 + 2 escenas, L7 13 + 34 en el bbox de prueba); 2026 es parcial (hasta agosto) y se publica con `span_days`/`n_obs` como aviso | lectura del archivo en el pod |
| D6 | grilla de la curva | 100 pasos entre la **primera y la última fecha con algún píxel despejado en la tesela** dentro de la ventana; interpolación lineal por píxel con extrapolación constante a los bordes; NaN con < 5 observaciones | réplica del cubo por parcela (`raw_series`: primera..última fecha del cubo); probada contra `interp_grid` píxel a píxel |
| D7 | teselas | 10 km ajustados a píxeles enteros (333 × 30 m = 9.990 m), origen en múltiplos, EPSG:32719 | mismo retículo que `dc.load`; los mosaicos no remuestrean |
| D8 | lectura Landsat | una sola carga por tesela para 1998–2026 (`red`, `nir`, `qa_pixel`; misma máscara QA por producto que `cube.load_window`), y cada año se corta en memoria | 27 años objetivo comparten 29 años de archivo; cargar por año leería 3× |
| D9 | retransformación | por semilla: clip al rango de entrenamiento en el espacio Yeo-Johnson, **Duan smearing con los residuos OOF del run block20 de la misma configuración** (`--oof-csv`, por defecto), clip al rango observado, y promedio de las 5 semillas | LCBD tiene λ ≈ −4.205 y escala 3,5e-6: sin clip un z fuera de rango explota. Medido en la compuerta (pod, 2026-09-02): la inversa simple pierde 0,25 de R² in-sample en TD₀ (0,50 → 0,74; λ = −2,2) y 0,04 en PD₀; sin smearing no se publica |
| D13 | facetas que se publican | **LCBD, PD₀ y TD₀**; las cuatro facetas ponderadas (PD₁, PD₂, TD₁, TD₂) se escriben en los GeoTIFF por tesela para diagnóstico pero no se mosaican ni se publican | su R² de block-CV es ≤ 0,04 o negativo (referencia recalculada del OOF: PD₁ −0,010, PD₂ +0,040, TD₁ −0,181, TD₂ −0,090): sin habilidad validada. Por defecto salvo indicación del autor |
| D10 | salida | un GeoTIFF por (tesela, año), 10 bandas float32: 7 facetas + `n_obs` + `span_days` + `native`; deflate, tiled; tags con todas las decisiones; `manifest.csv` reanudable | mosaico anual por `gdalbuildvrt` después |
| D11 | cómputo | CPU (no hay GPU). El piloto en el pod; la corrida completa en **dask-gateway, una tesela entera por tarea**. El pod hoy da 36 núcleos y **64 GiB** (no los 123 GB de la primera recon: encogió), y ese es el techo que el gateway rompe | §8 |
| D12 | despliegue | **piloto primero**: una tesela con 52 parcelas (Cauquenes, −36,0/−72,4), todos los años, medida en segundos/tesela y comparada con los valores de parcela; con eso se decide extensión final y resolución | elección del autor (Q3) |

## 3. Tamaño del problema (medido en el pod, MapBiomas 2024, decimado 10×)

277.933 km² de vegetación nativa entre 30° y 55,2°S ⇒ ~4,4·10⁸ píxeles de 30 m por año,
~1,2·10¹⁰ píxel-años, ~6·10¹⁰ pasadas de la CNN (5 semillas). Aproximadamente 2.800
teselas de 10 km. Lo que decide el calendario es la lectura de ~1.400 escenas por tesela
desde S3, no la CNN; el piloto mide ambas. Si el costo obliga, las alternativas —en este
orden— son: (a) restringir la extensión (Chile central primero), (b) grilla de salida de
90 m leyendo el píxel más cercano (misma señal de 30 m, 9× menos píxeles), (c) menos años.

## 4. Compuerta y piloto (qué corre el pod)

```bash
git pull --ff-only origin main
BIODIV_UNIFIED=1 BIODIV_CURVES=_raw100 python scripts/74_check_map_consistency.py
# solo si termina en ALL PASS:
python scripts/73_map_inference.py --bbox -72.50 -36.10 -72.35 -35.95 --years 2000-2026 \
    --area-m2 900 --stratum basal --mask mapbiomas --workers 8 --out results/maps --tag pilot_cauquenes
```

Antes de la corrida, la compuerta `scripts/74 --oof-csv ...` debe dar ALL PASS con los
checkpoints refit sobre la topografía corregida, y además el R² in-sample con smearing de
TD₀ debe alcanzar al menos el nivel que los checkpoints viejos daban sobre su propia
topografía (0,78): la compuerta por sí sola no distingue topografías (ALL PASS también con
la vieja), así que ese nivel es el criterio de que el refit recuperó lo perdido.

Reportar: segundos de carga por tesela, segundos por año, píxeles predichos por año,
rango de cada faceta, y la comparación píxel-de-parcela vs valor observado para las
parcelas de la tesela en su año de censo.

## 5. Corrección previa obligatoria: convención de aspecto (2026-09-02)

Verificado en el pod recalculando `terrain()` corregido sobre 30 parcelas Parcelas-CL y 10
Living Trees: en las filas `PCL_` de `topography_unified.parquet` `northness` y `eastness`
tienen correlación −1,000 con el recálculo (giro de 180°, las 1.082 filas) y `heat_load`
está espejado; en las filas `LT_` el acuerdo es 100 % (heat_load a 2e-4). El modelo final y
todos los runs de `docs/20` se entrenaron con esa mezcla. Decisión del autor: regenerar la
topografía de Parcelas-CL con el `scripts/03` corregido, reconstruir `topography_unified`,
re-correr block20, LLTO y barridos, refit `--all-data`, y actualizar `docs/20` y el paper.
Hasta que eso termine no se produce ningún mapa. Los valores antiguos quedan en git
(`b921daa`). Hallazgo colateral: `plots_unified.parquet` tiene `X`/`Y` (UTM 19S) en NaN
para las 2.020 filas `LT_`; se corrige en `scripts/51` con reproyección desde lon/lat.

## 6. Piloto medido (2026-09-02, tesela t18_600, Cauquenes)

Tesela de 9.990 m (333 x 333 = 110.889 px), 27 años, 8 workers locales, checkpoints
anteriores a la corrección de topografía (la corrida equivalente con los checkpoints
corregidos está en curso):

| magnitud | valor |
|---|---|
| carga Landsat 1998-2026 (1.811 fechas), una sola vez por tesela | 346 s |
| trabajo por año (curvas + inferencia + escritura) | mediana 31,2 s (21,9-37,7) |
| píxeles predichos por año (nativos con curva completa) | mediana 46.369 (31.901-53.173) |
| GeoTIFF por tesela-año, 10 bandas | 1,5 MB (41 MB los 27 años) |
| RAM | 6-8 GB de 123 disponibles |

Dos lecturas operativas. **La carga se amortiza**: es el 29 % de una corrida de 27 años y
se paga una vez por tesela. **El cuello de botella es el trabajo por año**, que corre en un
solo proceso mientras los ocho workers quedan ociosos: 0,673 ms por píxel-año. Extrapolado
a los 3,1·10⁸ píxeles nativos por año, la corrida completa 2000-2026 a 30 m son ~1.830 h en
serie (1.560 de inferencia + 270 de carga), es decir 57 h con 32 procesos o ~14 h en un
cluster de 128 núcleos. Alternativas medidas sobre la misma base: grilla de salida de 90 m
(441 h en serie), un año de cada tres (787 h), o restringir a 30-38°S (505 h).

**Subdispersión, declararla antes de que alguien lea un máximo como valor real.** En la
tesela piloto TD₀ llega a 42,8 contra un máximo observado de 84,0 en las parcelas, y la
mediana del mapa (6,8) queda bajo la mediana observada (9,3). El modelo comprime la cola
alta incluso con smearing; los mapas se interpretan como superficie relativa, no como
conteos absolutos de especies.

**La comparación con parcelas dentro de una tesela no es validación.** En el piloto caen
7-11 parcelas; la correlación de Spearman con ese n tiene error estándar cercano a 0,4. La
validación del modelo es la de la Sección 4 (bloques espaciales y LLTO); el contraste por
píxel es solo una lectura de coherencia.

## 7. Advertencias operativas

- `dask_gateway.Gateway().cluster_options()` imprime credenciales AWS STS y una
  contraseña de base de datos en su `repr`. No imprimirlo en notebooks, logs ni archivos
  versionados. `scripts/73` solo imprime `cluster.name` y el enlace del dashboard.
- Los pickles de `PowerTransformer` se escribieron con scikit-learn 1.3.1 y se leen con
  1.8.0 (`InconsistentVersionWarning`). La compuerta 1 de `scripts/74` (ida y vuelta del
  escalador) es la que dice si eso importa.
- `ck["rows"]` son las etiquetas de fila de la imagen serpentine, no las columnas de
  contexto; las columnas de contexto salen de `ck["ctx_preprocessor"].columns_`.

---

## 8. Dónde corre la inferencia, y por qué ahí (medido 2026-09-03)

La primera versión de `scripts/73 --jobs N` repartía teselas entre procesos hijos del pod y
cada hijo cargaba su tesela. Las tres mediciones de abajo desarmaron dos supuestos de ese
diseño y llevaron la corrida completa al gateway. Ninguna es una estimación.

### 8.1 La carga no escala con hilos: el GIL, no la red

Una tesela de 10 km, 1.812 fechas, span 1998-2026 (`logs/thread_scaling.log`,
`logs/load_scaling.log`):

| hilos | carga | vs 1 hilo | RSS pico |
|---|---|---|---|
| 1 (síncrono) | 1.892 s | 1,0x | — |
| 4 | 643 s | 2,9x | 2,4 GB |
| 8 | 562 s | 3,4x | 3,1 GB |
| 16 | 556 s | 3,4x | 3,7 GB |
| 24 | 559 s | 3,4x | 4,3 GB |
| 8 **procesos** | 304 s | 6,2x | — |

La carga es 96 % costo fijo (1.847 s + 765 s/Mpx): es abrir ~1.800 cabeceras COG, no mover
bytes. Pero **los hilos se aplanan en 4 y no pasan de 3,4x**, mientras ocho *procesos* dan
6,2x sobre la misma tesela: GDAL parsea esas cabeceras sosteniendo el GIL. La consecuencia de
diseño es directa —el paralelismo que paga es **un proceso por tesela**, con apenas 4 hilos
adentro— y corrige lo que este documento y el docstring de `scripts/73` afirmaban antes, que
los hilos escalaban casi linealmente.

### 8.2 Un cluster por tesela acelera la parte chica

Con 10 años objetivo, una tesela son ~300 s de carga contra ~1.100 s de CNN
(`--torch-threads 1`); con 27 años, ~300 s contra ~3.000 s. Un cluster dedicado a la carga
ataca entre el 9 % y el 21 % del trabajo y deja sus workers ociosos el resto —la misma
patología que el piloto ya mostró (§6)—. Por eso el gateway recibe la **tesela entera**
(carga + curvas + CNN + escritura), no la carga.

### 8.3 Qué ve un worker del gateway (`logs/gw_probe.log`)

| | worker |
|---|---|
| recursos por defecto | 16 núcleos, 28 GB |
| home / repo / MapBiomas | **no visibles** |
| `torch` | 2.12.0+cpu |
| índice ODC | accesible (43 productos; el gateway inyecta `DB_*`) |
| versiones | numpy 2.3.5, pandas 3.0.3, rasterio 1.5.0, sklearn 1.8.0, datacube 1.9.18 |

La imagen del worker es **idéntica a la del pod**, versión por versión. Eso cierra la
advertencia de §7 sobre los pickles de `PowerTransformer` escritos con scikit-learn 1.3.1:
se leen en el worker exactamente como en el pod, y no hay una segunda combinación de
versiones que auditar.

Lo que el worker no tiene se le manda: el paquete `biodiv` como zip (`Client.upload_file`)
—con `scripts/03_extract_topography.py` adentro como `biodiv/_terrain_src.py`, porque
`load_terrain` lee `terrain()` de ese script por ruta para que siga siendo la única fuente de
las derivadas topográficas, y una carga por ruta no alcanza el interior de un zip; la copia se
reconstruye del script vivo en cada corrida, así que no pueden divergir dentro de una—,
los cinco checkpoints como bytes en un `Payload` difundido una vez —no viajando con cada una
de las ~5.900 tareas—, MapBiomas leído del bucket de scratch (`BIODIV_MAPBIOMAS_DIR`, ventana
de 557×557 en 0,68 s porque los rasters son tiled 512×512 con overviews) y los GeoTIFF
escritos de vuelta al scratch. El worker devuelve solo la fila del manifest.

**El bucle por tesela vive en `src/biodiv/maptask.py`, no en el script.** La compuerta de
`scripts/74` produce mapas solo si el camino de mapa reproduce exactamente el de
entrenamiento, y esa prueba no vale nada si el código que la compuerta revisó no es el que
corrieron los workers. Driver local y worker importan la misma función.

**El scratch se borra a los 30 días.** Los mosaicos anuales de las tres facetas publicables
(D13: LCBD, PD₀, TD₀) hay que generarlos y bajarlos dentro de esa ventana; los GeoTIFF por
tesela-año con las siete facetas son intermedios y no sobreviven a propósito.

### 8.4 Presupuesto medido, y la corrida que se lanzó (2026-09-03)

Ocho teselas repartidas de 30° a 55°S, 10 años, 8 workers de 2 núcleos
(`logs/calib_gateway.log`), medianas por tesela:

| pieza | mediana | rango |
|---|---|---|
| carga Landsat 1998-2026, una vez por tesela | **1.102 s** | 553-1.777 |
| trabajo por año | **28 s** | 1,6-56 |
| inferencia por píxel-año | 0,52 ms | — |

**La carga es el ~80 % de la tesela, no la CNN.** Eso invierte el supuesto de §6, que salía
de un piloto con torch sobre los 36 núcleos del pod; un worker recibe 2. La consecuencia es
la que decidió el alcance: la carga se paga una vez cubra la tesela 10 años o 27, así que
**27 años cuestan 35 % más que 10, no 2,7x** (3.059 contra 2.268 horas-tesela). Recortar años
ahorra poco y cuesta casi toda la serie temporal.

Cobertura nativa por tesela, sobre la misma grilla decimada de `scripts/76`
(`results/figures/tiles_native_10km_cover.csv`): mediana 1.532 píxeles de ~3.000 posibles.
La sospecha de que muchas teselas retenidas estaban casi vacías **era falsa** —la
distribución está cargada hacia teselas llenas—; t16_455, que pagó 1.102 s de carga para 271
píxeles, es la excepción. Descartar las de menos de 20 píxeles nativos saca 123 teselas
(2,1 %), ahorra ~38 h y pierde 0,01 % del área: se aplicó, y el resto no, porque a partir de
ahí ya se cambia cobertura por tiempo.

Lo que corre (decisiones del autor, 2026-09-03):

```bash
python scripts/73_map_inference.py \
    --tiles-file results/figures/tiles_native_10km_run.csv --years 2000-2026 \
    --area-m2 900 --stratum basal --mask mapbiomas \
    --gw-workers 128 --worker-cores 2 --worker-memory 8 --worker-threads 1 \
    --load-threads 4 --dest s3://<scratch>/biodiv/maps/chile_30m_2000_2026 \
    --mapbiomas-dir s3://<scratch>/biodiv/MapBiomas \
    --out results/maps --tag chile_full_30m --resume
```

5.769 teselas, 27 años, ~191 GB de salida en el scratch. Con los 128 workers concedidos en
pleno es ~1 día; con menos, proporcionalmente más. **Pendiente y con plazo:** los mosaicos
anuales de LCBD, PD₀ y TD₀ hay que construirlos y bajarlos antes de que el scratch expire a
los 30 días.

### 8.5 La corrida se detiene sola, y por qué (2026-09-04)

El contenedor se reinició durante la noche y mató las dos mitades: los 14 procesos del pod
—sus logs terminan a mitad de tesela, sin errores— y el driver del gateway. `setsid` protege
de que termine la sesión que lanzó el trabajo, que es de lo que protegió dos veces; **no
protege de que se reinicie el contenedor**, y nada que corra dentro de él puede hacerlo.

La causa encaja con un **culler por inactividad**: última actividad interactiva a las 20:07,
contenedor levantado de nuevo a las 01:51 cuando algo volvió a abrirlo. No está confirmado
directamente —el hub responde 403 en `/hub/api/services` y `/info` con el token del pod— pero
sí lo está el desajuste de fondo:

> El servidor reporta actividad al hub (`JUPYTERHUB_ACTIVITY_URL`), y JupyterHub la mide por
> **kernels y peticiones HTTP, no por CPU**. Quince procesos saturando 36 núcleos son
> invisibles para esa métrica: a ojos del hub este servidor estaba ocioso mientras corría a
> plena carga.

Decisión del autor (2026-09-04): **aceptarlo y relanzar**, en vez de registrar actividad
artificial contra `JUPYTERHUB_ACTIVITY_URL` —que sería derrotar a propósito un mecanismo de
reparto en infraestructura compartida— o de pedir una exención a los admins. La consecuencia
para el calendario hay que declararla: 4,2 días de cómputo se convierten en 7-10 días de
calendario si se pierden ~6 h por noche.

Lo que se relanza, una sola línea, idempotente (una mitad ya viva se deja en paz, porque
lanzarla dos veces correría las mismas teselas en dos sitios y competiría por el manifiesto):

```bash
cd ~/temp/Biodiversity_Chile && scripts/start_maps.sh
```

`scripts/run_maps_supervised.sh <gateway|pod>` es el supervisor por mitad: cuenta lo que falta
como las teselas sin sus 27 años escritos y relanza con `--resume` hasta que no queda ninguna.
Un relanzamiento solo cuesta las teselas que estaban en vuelo.

### 8.6 El cambio a 1D-CNN no mueve el presupuesto (medido 2026-09-09)

El modelo desplegado pasó del 2D-CNN a la 1D-CNN (`docs/20` §9.7). Como la 1D no pliega la
curva en imagen, cabía esperar que el trabajo por año cayera lo suficiente como para que la
carga volviera a dominar y §8.4 quedara **sobreestimada**. Se midió y **no ocurre**: §8.4 se
queda como está.

Lo único que cambia entre familias es `model_inputs` + `predict` —la carga de Landsat y el
`_emit` del GeoTIFF son el mismo código—, así que se cronometró ese término solo, en una
misma máquina y con el tamaño real de una tesela (40.190 píxeles predichos, 5 semillas):

| familia | forma de entrada | `model_inputs` | `predict` | total | ms/píxel |
|---|---|---:|---:|---:|---:|
| C1D | `(40190, 1, 100)` | 0,000 s | 26,00 s | **26,00 s** | 0,647 |
| C2D | `(40190, 1, 10, 10)` | 0,012 s | 27,58 s | **27,59 s** | 0,687 |

**La 1D es 5,8 % más barata, y la transformada serpentina no costaba nada**: 12 ms por año
sobre 37.500. El ahorro viene del forward, no del plegado. Trasladado a la línea base de
§8.4 *en su propio hardware*, que es la única forma válida de compararlo: el trabajo por año
pasa de 28 s a 26,4 s, la tesela de 27 años de 1.858 s a 1.815 s, y el total de Chile de
3.059 a ~2.988 horas-tesela — **un 2,3 %**, muy por debajo de la dispersión entre teselas de
la propia §8.4 (carga 553–1.777 s).

Corolario que corrige una lectura ingenua de §8.4: **el forward es el ~69 % del trabajo por
año** (26,0 s de 37,5 s; el resto es `year_curves`, la máscara MapBiomas y la escritura del
GeoTIFF). El «la carga es el ~80 %» de §8.4 es sobre **10 años y por tesela completa**,
porque la carga se paga una sola vez; dentro del término por año la CNN nunca fue
despreciable.

Piloto de verificación (`results/maps/pilot_c1d/`, tesela `t2_446`, 27/27 años ok, compuerta
`scripts/74` en ALL PASS con errores de 0,00e+00 en contexto, entradas y forward):

| pieza | pod, 8 núcleos, `--torch-threads 4` | §8.4, gateway, 2 núcleos |
|---|---:|---:|
| carga Landsat 1998–2026 | 294 s | 1.102 s |
| trabajo por año (mediana) | 37,5 s | 28 s |
| ms por píxel-año | 0,934 | 0,52 |

**Estos absolutos no entran a §8.4 y no son comparables con ella.** La línea base se midió en
workers de gateway de 2 núcleos con torch pinneado a 2 (`logs/calib_gateway.log`); el piloto
corrió en el pod con 8 núcleos y 4 hilos de torch. Que el ms/píxel-año del pod sea *peor*
(0,934 contra 0,52) lo demuestra: son máquinas distintas, no modelos distintos. Sustituir una
serie por la otra habría atribuido al modelo una diferencia que es de hardware.

#### `--worker-cores 2` deja de ser elección y pasa a ser medición

Dado que el forward es el ~69 % del trabajo por año, cuánto escala con hilos de torch decide
el reparto de núcleos. Medido sobre el C1D desplegado, 5 semillas:

| hilos | ms/píxel | speedup | eficiencia |
|---:|---:|---:|---:|
| 1 | 1,512 | 1,00× | 100 % |
| 2 | 0,848 | 1,78× | **89 %** |
| 4 | 0,600 | 2,52× | 63 % |
| 8 | 0,450 | 3,36× | 42 % |

El escalado es marcadamente sublineal. En la ruta de gateway torch queda pinneado a
`--worker-cores` (`scripts/73_map_inference.py:387`), de modo que la corrida de §8.4 usa 2
hilos: **el punto eficiente**. Con un presupuesto fijo de núcleos conviene repartirlos en
muchos workers flacos, no en pocos gordos —a 8 hilos se desperdicia el 58 % de cada núcleo
añadido—, así que `--worker-cores 2` se mantiene, ahora con una medición detrás.

### 8.7 El smearing era un tercio del año-tesela (medido y corregido 2026-09-09)

Perfilando por etapas un año-tesela real (`t18_600`, 38.934 píxeles predichos, 5 semillas)
aparece un costo que §8.4 no podía ver, porque cronometra "trabajo por año" como un bloque:

| etapa | s | % |
|---|---:|---:|
| `predict_scaled` (forward torch) | 21,60 | 63,2 % |
| **retransformación + smearing** | **10,83** | **31,7 %** |
| `year_curves` | 1,52 | 4,4 % |
| `native_mask` + `model_inputs` + GeoTIFF | 0,25 | 0,7 % |

`inverse_with_smearing` hace 128 llamadas a `scaler.inverse_transform`, una por draw de Duan,
repetidas por semilla: **640 inversas por año-tesela**. Las salidas obvias no sirven, y se
midieron antes de descartarlas: apilar los draws en una sola llamada a sklearn no gana nada
(2,16 contra 2,20 s), vectorizar la inversa Yeo-Johnson en numpy monohilo es *más lenta*
(0,86x), y en torch da 1,43x con 8 hilos pero **0,69x con 2**, que es lo que tiene un worker
de gateway. El costo es la aritmética —`np.power` sobre 34,9 M elementos float64—, no el
overhead de sklearn.

Lo que sí sirve es la estructura: `draws` es un escalar por (draw, target) difundido sobre
todas las filas, y el clip es por columna, así que para un target fijo el estimador entero es
una **función monótona de una sola variable** sobre el dominio acotado `[lo_s, hi_s]`. Se
tabula una vez por semilla (`targets.build_smearing_table`) y se interpola.

A/B sobre la misma tesela, exacto contra tabla, tres años:

| | exacto | tabla |
|---|---:|---:|
| primer año (incluye construir la tabla) | 47,1 s | 42,6 s |
| años siguientes | 40,8 s | **28,1 s (1,45x)** |

Peor error relativo 1,7e-5, siempre en TD₀ —2e-4 especies sobre ~7—, y `n_obs`, `span_days` y
`native` bitwise idénticas. La compuerta `scripts/74` sigue en ALL PASS con la misma tabla de
R² y los mismos gaps `[0,053, 0,03]`.

**La tabla es solo del camino de mapas.** `inverse_with_smearing` también corre en
`dl_runner.py` y en los baselines, de donde salen todos los R² publicados (§9.3, §9.5 y §9.7
de `docs/20`); ahí el smearing corre sobre 3.102 parcelas y no cuesta nada, así que la ruta
exacta sigue siendo la de la validación cruzada y esos números no se mueven. `--smearing-exact`
fuerza la ruta vieja en los mapas.

El tamaño de grilla (`SMEARING_GRID`) se dimensiona contra el **costo de construir**, no
contra el error, que sobra en todo el rango: construir cuesta una pasada de los 128 draws
sobre M filas, así que un M cercano al número de píxeles de una tesela hace que la tabla
cueste tanto como el año que reemplaza —a 65.536 una tesela de un año midió 0,87x, más lenta
que el camino exacto.

### 8.8 La interpolación pagaba por toda la tesela, y pedía 2,3 GiB (medido y corregido 2026-09-10)

`run_tile` interpolaba los 110.889 píxeles de la tesela y *después* descartaba los que la
máscara de MapBiomas deja fuera. El reordenamiento —máscara y ventana antes de interpolar—
se apoya en una identidad que queda verificada en un test:

> `np.isfinite(curves).all(axis=1)` es **exactamente** `n_obs >= MIN_OBS`. Una curva vuelve
> NaN sólo si el píxel no tuvo ninguna observación despejada, y entonces lo es en todos los
> puntos de la grilla, nunca en algunos.

Así que qué píxeles vale la pena correr se sabe **antes** de interpolar. Lo que no se puede
diferir es `lo`/`hi`: la grilla abarca la primera y la última fecha con algún píxel despejado
en **toda** la tesela (la convención del cubo de parcelas, D6), así que `year_window` calcula
ese tramo sobre todos los píxeles, de la misma pasada de `isfinite` que produce `n_obs`.
Enmascarar antes de eso movería la grilla en silencio.

**El ahorro en tiempo depende de la tesela y no se puede presupuestar.** De las cuatro
teselas medidas, tres son 96 % nativas (105.987 de 110.889 px) y ahí no ahorra casi nada;
t18_529 predice 6.940 px/año y ahí ahorra ~94 %. Las teselas de §6 y §8.7 están en el medio
(35-42 % de píxeles predichos). Nada de esto entra al presupuesto de §8.4.

**Lo que sí es incondicional es la memoria, y esa es la razón del cambio:**

| | pico RSS del camino de tesela, 10 km |
|---|---:|
| antes | 2.345 MiB |
| después | **294 MiB** |

Tres cambios, ninguno aproximado: índices `int32` en vez de los `int64` que numpy elegiría
—son números de fila en una ventana de unos cientos de fechas, y los cuatro arreglos `(T, N)`
eran 0,93 GB del pico—, `np.copyto(..., where=)` en vez de cinco `np.where` encadenados que
alocaban un `(G, N)` float64 nuevo por rama, y un parámetro `block` que interpola por bloques
de columnas (`TileConfig.interp_block`, 20.000 por defecto). El transitorio de
`interp_common_grid` es ~20x su propia entrada, y es lo que acota cuántas teselas caben en un
pod.

Las dos consecuencias operativas: `--jobs 6/7` entra en los 30 GiB del pod (§8.9), y las
teselas de 20 km dejan de ser imposibles —ahí el interp sin bloquear pediría ~9,1 GiB por
proceso—.

**Bit a bit idéntico, no aproximado.** Verificado sobre t18_600 en 2005/2015/2024 contra la
salida de `main`: 3 rásters × 10 bandas con `np.array_equal(equal_nan=True)`, manifiesto
incluido. En tests, además, para subconjuntos de columnas, para todo tamaño de bloque
(incluidos los que no dividen N) y para tiempos desordenados.

**Caché de teselas de desarrollo (`BIODIV_TILE_CACHE`), que Argo no define y por lo tanto
nunca usa.** En producción cada tesela se visita una sola vez y la caché sería 0,7 GB de
escritura para nada. Existe porque iterar sobre el bucle de años costaba la carga entera de
Landsat cada vez —714 s medidos en el tramo 1998-2026—; con la caché la misma corrida baja a
**1m54** y da salida idéntica. Los arreglos se mapean en memoria, así que un acierto no cuesta
copia y el bucle de años pagina sólo la ventana que toca. La escritura es atómica
(`tmp.replace(d)`) y la geometría se re-verifica contra la tesela, de modo que una caché
escrita para otra grilla no puede ser recogida por una tesela del mismo nombre.

Esa caché es, a escala de una tesela, el mismo producto que propone `docs/24`: la medición de
714 s → 0 s es la evidencia más directa que hay de cuánto vale materializar la carga.

### 8.9 El pod de Argo usaba un séptimo de sí mismo (medido y corregido 2026-09-10)

El pod pide `cpu: '7'` y corría `--jobs 1`: seis de sus siete núcleos ociosos. **Ninguna de
las dos mitades de una tesela puede usar el pod entero por sí sola**, y las dos razones ya
estaban medidas en este documento sin que se sacara la consecuencia:

- la carga está topada por el GIL en **~0,7 de un núcleo** (§8.1: GDAL lo retiene mientras
  parsea las cabeceras COG; medido instantáneo, no promedio de vida del proceso);
- la eficiencia por hilo de torch cae a **63 % con cuatro hilos** (§8.6).

Lo que sí escala es un proceso monohilo por núcleo, cada uno con una tesela entera, porque la
espera de S3 de una se solapa con la CNN de otra. Medido en el pod de Jupyter de 7 núcleos
(`taskset -c 0-6`, 7 teselas, 3 años):

| `--jobs` | teselas/h | vs `--jobs 4` |
|---:|---:|---:|
| 4 | 27,4 | 1,00x |
| 7 | **42,6** | 1,56x |

1,75x los procesos comprando 1,56x el trabajo: **89 % de eficiencia, sin señales de pared**.
Contra el `--jobs 1 --workers 0` medido antes (2.044 s para 4 teselas × 5 años), el cambio
vale **~1,6x end to end**.

**Queda en 6 y no en 7 a propósito.** En el pod de desarrollo `limit` no está puesto y
`taskset` sólo reparte, pero en Argo `limit == request`: a utilización exactamente igual a la
cuota el throttling de CFS frena todo el cgroup por períodos de 100 ms, y eso cuesta más que
el núcleo que se deja libre. Antes de subir a 7 hay que mirar `/sys/fs/cgroup/cpu.stat`
(`nr_throttled`) en un chunk real; el pod de desarrollo no puede responderlo porque no tiene
cuota.

**La memoria no es el límite**, y lo es gracias a §8.8: 1,24 GB por proceso medidos en un
tramo de 400 fechas, ~1,8 GB en el tramo de producción (`obs` crece con las fechas), o sea
~11 GB de los 30 Gi del pod.

**El chunk se dimensiona contra el deadline, no contra la tesela.** A ~32 min por tesela por
proceso (27 años), una ronda de seis dura ~32 min:

| teselas/pod | rondas | duración | ¿entra en 7.200 s? |
|---:|---:|---:|---|
| 18 (90/5) | 3 | ~96 min | **sí**, ~20 % de margen |
| 24 | 4 | ~128 min | no, los pods mueren en el deadline |

Las teselas por pod tienen que ser **múltiplo de 6** o la última ronda corre con procesos
ociosos. El default pasa a `-p limit-tiles=90 -p num-chunks=5`. La corrida del 2026-09-08 se
estrelló contra esta misma pared con la aritmética vieja (50 teselas / 5 chunks, 10 por pod,
~150 min sobre `--jobs 1`).

`parallelism` no se toca: un solo cambio por vez, para poder atribuir el resultado.

**Dos consecuencias de correr seis hijos en vez de uno**, ambas de corrección y no de
rendimiento. `configure_s3_access` registra una config rio **por defecto** que no cruza a un
subproceso, así que las variables de entorno de requester-pays (`AWS_REQUEST_PAYER`) dejan de
ser un cinturón redundante y pasan a ser la única cosa que sostiene el acceso a
`usgs-landsat` en los seis hijos. Y `--mapbiomas-dir` dejó de ser obligatorio
(`mapbiomas.rasters_dir` ya apunta por defecto a ese prefijo); se deja escrito como
declaración explícita. Corolario documental: los rásters de MapBiomas **no** se copian al pod
—son COGs leídos por ventana desde S3, 0,79 s la máscara de una tesela de 10 km—, así que
copiar 3,6 GB a cada pod no compraría nada.
