# Cubo kNDVI materializado: plan de prueba y de escalado

**Estado: no empezado. No hay código escrito.** Este documento es un plan y sus números son,
salvo donde se diga lo contrario, *derivados* de las mediciones de `docs/21` §8, no medidos
sobre un cubo Zarr que todavía no existe. La Fase 0 existe justamente para reemplazar esas
derivaciones por mediciones.

## 1. Qué se propone

Construir el cubo kNDVI **una sola vez** como Zarr, guardarlo en S3, y correr
`scripts/73_map_inference.py` contra eso en vez de leer ~1.800 ventanas de COG de
`usgs-landsat` por tesela.

Hay un precedente medido a escala de una tesela, y no es una analogía: `BIODIV_TILE_CACHE`
(`docs/21` §8.8) ya **es** un producto pre-materializado en pequeño —un `.npy` mapeado en
memoria con el kNDVI de una tesela— y llevó la carga de **714 s a 0 s**, con salida idéntica.
La pregunta de este documento no es si materializar la carga sirve; es si se puede hacer para
las 5.769 teselas sin romper la reproducibilidad ni gastar más de lo que ahorra.

## 2. Por qué, con la aritmética a la vista

Sobre la línea base de gateway de §8.4, actualizada por §8.6 (1D-CNN) y §8.7 (tabla de
smearing) — **derivado, no medido como tesela completa**:

| pieza | s por tesela (27 años) | % |
|---|---:|---:|
| **carga Landsat** | **1.102** | **69,2 %** |
| forward torch (5 semillas) | 378 | 23,7 % |
| resto (curvas, smearing tabulado, máscara, GeoTIFF) | 114 | 7,1 % |
| total | 1.594 | 100 % |

La carga subió de share, no de costo: es fija en 1.102 s, y §8.6 y §8.7 abarataron todo lo
demás. Cuatro consecuencias, en el orden que importa:

1. **Las re-corridas salen casi gratis.** Construir cuesta aproximadamente una fase de carga
   completa (5.769 × 1.102 s ≈ **1.766 horas-proceso**) y la ahorra en cada corrida
   posterior. La corrida ya se reinició varias veces (§8.5, y las pérdidas de 3.434 y 3.320
   teselas por `scheduler-connection-lost`), así que el punto de equilibrio ya quedó atrás.
2. **Elimina clases enteras de fallo:** credenciales STS que expiran a mitad de lectura,
   `AccessDenied` de requester-pays, varianza de latencia de S3, y schedulers que se caen
   durante cargas de ~700 s.
3. **La lectura vuelve a escalar con hilos.** El techo de 3,4x de §8.1 es GDAL sosteniendo el
   GIL para parsear cabeceras COG; blosc/zstd lo suelta. Ese es el único cambio que ataca la
   causa medida en §8.1 en vez de rodearla.
4. **Da vuelta el caso de la GPU.** Con la carga cerca de cero, el forward pasa de **~24 % a
   ~77 %** de la tesela. Una GPU deja de ser una palanca sobre un cuarto del trabajo y pasa a
   serlo sobre tres cuartos. Por eso la caché va primero y la GPU después: al revés compra
   mucho menos.

**Dimensionamiento (estimación, la Fase 0 lo mide).** 5.769 teselas × 110.889 px × ~1.811
fechas × 4 B float32 ≈ **4,6 TB** en crudo. El arreglo es mayoritariamente NaN —la fracción
exacta es una de las cosas que hay que medir, no suponer— así que con zstd el orden esperado
es de **~1 TB**, pero ese número no está verificado y el plan no debe apoyarse en él hasta
que la Fase 0 lo confirme.

## 3. Fase 0 — consistencia a pequeña escala, antes de cualquier decisión de escalado

**Esto es la compuerta y va primero.** El riesgo dominante no es de costo, es de
reproducibilidad: el Zarr tiene que reproducir `dc.load` **exactamente**, o se rompen la
compuerta de `scripts/74` y la identidad bit a bit que §8.8 acaba de establecer.

Lo que tiene que coincidir, y cada uno es una forma distinta de equivocarse en silencio:

| tiene que coincidir | por qué se rompe si no |
|---|---|
| grilla de salida (CRS, origen, resolución, forma) | D7: los mosaicos no remuestrean |
| `group_by="solar_day"` | dos escenas del mismo día solar se funden o no |
| máscara QA por producto (`cube.load_window`) | cambia `n_obs`, y con eso qué píxeles se predicen |
| reproyección por vecino más cercano | cualquier otro remuestreo inventa valores |
| orden de empate dentro del mismo día solar | el `argsort` estable de `interp_common_grid` lo hace **observable** |

Ese último es el que se pasa por alto. `interp_common_grid` ordena por tiempo con
`kind="stable"`, así que dos observaciones de la misma fecha conservan el orden en que
llegaron; si el Zarr las guarda en otro orden, las curvas cambian aunque el conjunto de datos
sea el mismo.

**La prueba es barata y es la única que hay que correr antes de decidir nada más:**

1. materializar **una** tesela (t18_600, que es la que tiene línea base en §6, §8.7 y §8.8);
2. correr `scripts/73` contra el Zarr y contra `dc.load`, mismos años;
3. diferencia de rásters con `np.array_equal(equal_nan=True)`, 10 bandas, manifiesto incluido
   — el mismo protocolo que validó §8.8;
4. `scripts/74` en ALL PASS.

Salidas de la Fase 0, además del veredicto: fracción real de NaN, tamaño comprimido de una
tesela, tiempo de construcción de una tesela, y tiempo de lectura desde el Zarr con 1, 4 y 8
hilos —que es lo que confirma o refuta el punto 3 de §2—.

Si la igualdad bit a bit no se alcanza, la decisión que sigue es **declarar una tolerancia o
abandonar**, y esa es del autor, no del código.

**Nota de diseño, ya decidida:** el chunking va a lo largo de todo el eje temporal con bloques
espaciales modestos, porque el patrón de acceso es "toda la serie de tiempo de una región
espacial". Y **no** se guardan curvas pre-computadas en vez de kNDVI: hornean la interpolación
y la convención de ventana (D6) dentro del producto, y no ocupan menos.

## 4. La tensión de diseño: el cluster de dask estorba al procesamiento

La observación que ordena todo el escalado es que **las dos fases quieren máquinas
distintas**, y no un poco distintas:

| | construir la caché | inferir |
|---|---|---|
| límite | I/O, topado por el GIL a ~0,7 de núcleo (§8.1) | cómputo (forward torch) |
| forma que quiere | muchos procesos flacos, mucha CPU por GB | pocos procesos, o una GPU con vCPU al lado |
| paralelismo que paga | procesos, no hilos (6,2x contra 3,4x, §8.1) | hilos de torch hasta ~2 (89 % de eficiencia, §8.6) |
| dask | ayuda | **estorba** |

Esa última fila es el punto. Un cluster de dask dimensionado para saturar la lectura deja sus
workers ociosos durante la inferencia —es exactamente la patología de §6 y §8.2, por la que el
gateway recibe la tesela entera y no la carga—. Materializar el cubo **separa** las dos fases y
por eso permite, por primera vez, darle a cada una la forma que quiere. Las tres opciones de
abajo se diferencian en *dónde* se pone ese corte.

## 5. Opciones de escalado

### Opción A — dos etapas, pods distintos

Tier A: pods con alta relación CPU/RAM cargan teselas y empujan Zarr a S3. Tier B: pods
optimizados para procesamiento (o con GPU) leen de S3 y hacen el análisis.

- **A favor:** cada tier se dimensiona a su límite real; los tiers escalan
  independientemente; un fallo en A no toca a B; encaja con lo que Argo ya sabe hacer, y la
  idempotencia ya está resuelta —`scripts/argo/tile_progress.py` comprueba qué hay en S3, y
  comprobar Zarr en vez de GeoTIFF es el mismo patrón—.
- **En contra:** cada tesela cruza S3 dos veces (escribir en A, leer en B), lo que agrega
  costo y latencia de red que la Opción B no paga; hay que operar dos definiciones de pod.
- **Dependencia:** la más simple es estrictamente bifásica (todo A, después todo B), no
  pipelining por tesela. Vale la pena resistir la tentación del pipelining hasta que A haya
  corrido entero una vez.

### Opción B — un solo pod, corte duro en el medio

Un pod escribe el Zarr a disco local, **mata el cluster de dask**, y reasigna sus recursos al
análisis.

- **A favor:** no hay ida y vuelta por S3 para el intermedio; una sola definición de pod; el
  dato está en disco local, que es la configuración que midió 714 s → 0 s en §8.8.
- **En contra:** el pod tiene que estar dimensionado para el **máximo** de las dos fases, no
  para cada una, así que durante la fase de análisis sobra RAM y durante la carga sobra poco;
  el intermedio no sobrevive al pod, así que un pod que muere —y §8.5 documenta que
  mueren— pierde las dos fases y no sólo una; y no hay reutilización entre corridas, que es el
  beneficio nº 1 de §2, salvo que igual se suba a S3.
- **Requisito:** el corte tiene que ser *duro* —cluster cerrado, memoria liberada, verificado—
  antes de que empiece el análisis, o se paga lo peor de las dos formas a la vez.

### Opción C — construir la caché en un cluster remoto con workers spot

- **A favor:** el grafo es mucho más amable que el de la inferencia que perdió 3.434 teselas:
  ~5.769 tareas gruesas independientes, sin payload compartido (sin checkpoints, sin
  residuos OOF, sin `y_train`), resultados de casi cero bytes (escribir a S3, devolver un
  dict de estado) y **sin anidamiento** — siempre que cada tarea llame a `dc.load`
  sincrónicamente dentro del worker en vez de construir un grafo lazy que se reenvía al
  scheduler que la está corriendo. Eso, y no el número de tareas, es lo que rompió las
  corridas anteriores. Spot encaja porque cada tesela es idempotente: un worker interrumpido
  cuesta una tesela.
- **Hay que comprobar algo primero:** las sondas de capacidad registraron **~5 worker pods
  concedidos independientemente de lo que se pidiera** (`logs/capacity_probe*.log`, y
  `logs/chile_full_30m.log`: "5 workers up (10 cores granted of 256 asked)"). Si esa es una
  cuota permanente, la ruta de cluster topa cerca de 40 núcleos y Argo gana por default; si
  fue capacidad transitoria, dask-on-spot es probablemente el camino más barato.

### Cómo elegir

No hace falta decidir ahora, y no conviene: la Fase 0 produce el número que separa A de B
—cuánto pesa una tesela comprimida y cuánto tarda en subir y bajar de S3—, y la sonda de
capacidad separa C de las otras dos. El orden es Fase 0 → sonda de capacidad → elección.

## 6. La GPU, que es una prueba y no una construcción

No necesita el Zarr. `BIODIV_TILE_CACHE` ya deja una tesela con la carga en cero, y `--device`
(`scripts/73_map_inference.py:132`), `TileConfig.device` (`src/biodiv/maptask.py:153`) y el
argumento de dispositivo de `FacetEnsemble` ya existen. Así que la prueba es: correr
`scripts/73` contra una tesela cacheada con `--device cuda` en una máquina con GPU, y
compararla con la misma tesela en CPU.

**Qué esperar, para no leer mal el resultado.** El modelo son 14.535 parámetros de Conv1d
separable en profundidad con `padding_mode="circular"`: intensidad aritmética muy baja, una
forma que las GPUs manejan mal. Lo más probable es quedar limitado por lanzamiento de kernels,
no por FLOPs. El batch hoy es 8.192; puede hacer falta subirlo, o agrupar entre teselas, para
mantener la GPU alimentada.

**La bit-exactitud no aplica aquí.** El orden de las operaciones de punto flotante difiere
entre CPU y GPU, así que la diferencia de rásters de §8.8 no va a dar cero y necesita una
**tolerancia declarada** en vez de igualdad. Conviene fijarla antes de mirar el resultado.

## 7. Secuencia

1. **Fase 0**, una tesela: consistencia bit a bit + fracción de NaN + tamaño + escalado de
   lectura por hilos. Compuerta.
2. **Sonda de capacidad**: ¿los ~5 workers son cuota o fueron circunstancia?
3. **Elegir A, B o C** con (1) y (2) en la mano.
4. **Prueba de GPU** sobre una tesela cacheada, con tolerancia declarada. Independiente de
   (1)-(3); se puede correr en paralelo.
5. Construir.

## Referencias

- `docs/21` §8.1 (el GIL y el techo de 3,4x), §8.2 (por qué el gateway recibe la tesela
  entera), §8.4 (presupuesto), §8.5 (la corrida se detiene sola), §8.6 (1D-CNN y escalado de
  hilos de torch), §8.7 (tabla de smearing), §8.8 (memoria de la interpolación y
  `BIODIV_TILE_CACHE`), §8.9 (`--jobs 6`).
- `scripts/argo/tile_progress.py` — el patrón de idempotencia contra S3 que las opciones A y C
  reutilizan.
