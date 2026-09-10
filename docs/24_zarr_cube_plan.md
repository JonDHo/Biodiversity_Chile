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

**Dimensionamiento — ya no es estimación, se midió (2026-09-11, `scripts/bench/bench_zarr.py`,
`logs/bench_zarr.log`).** Sobre t18_600, tramo 2018-2020, 238 fechas, 333 × 333 px:

| | medido | lo que decía este documento |
|---|---:|---:|
| fracción NaN | **40,3 %** | ~55 % |
| razón de compresión (blosc-zstd) | **2,1x** | implícita ~4x |
| por tesela al tramo completo | **~381 MB** | ~180 MB |
| **total de las 5.769 teselas** | **~2,2 TB** | ~1 TB |

**El cubo es más del doble de lo que este plan suponía.** El arreglo es bastante menos disperso
de lo que se había asumido, y ahí se va la diferencia entera. Ni el nivel de zstd (3 contra 5:
50,8 y 49,8 MB) ni el tamaño de chunk (de la tesela entera a bloques de 64 px: 49,8 a 50,1 MB)
mueven la aguja, así que no hay nada que afinar por ese lado. A precios de S3 son ~50 USD al mes,
o sea que el tamaño no es un impedimento — pero el número había que corregirlo.

Vale seguir notando que el cubo es **más chico que lo que se transfiere hoy**: el camino actual
trae `red` + `nir` + `qa_pixel` como uint16 (~6 B/px/fecha) para derivar un kNDVI de 4 B.

**Corrección al punto 3 de arriba: el argumento del GIL era el equivocado, y la conclusión es
mucho mejor de lo que decía.** Se midió la descompresión con el store abierto una sola vez, para
no cronometrar el costo fijo de abrirlo:

| hilos | s | speedup |
|---:|---:|---:|
| 1 | 0,174 | 1,00x |
| 2 | 0,112 | 1,55x |
| 4 | 0,078 | 2,24x |
| 8 | 0,066 | **2,63x** |

O sea que **blosc tampoco escala linealmente, y ni siquiera le gana al 3,4x de GDAL** de §8.1.
El punto 3 tal como estaba escrito es falso. Pero es irrelevante, y por la mejor de las razones:
al tramo de producción esa descompresión son **~1,3 s a un solo hilo**, contra los **~700 s** que
cuesta hoy abrir ~5.970 cabeceras COG. La ganancia no es una curva de escalado mejor; es que **no
queda prácticamente nada que escalar** — del orden de 500x sobre el término de carga, antes de
sumarle la transferencia desde S3.

Con eso, el presupuesto por tesela de §2 pasa de ~1.915 s a **~1.218 s (−36 %)**, y el forward
queda en el **77 %** de la tesela, que es exactamente el régimen en el que la GPU pasa a ser una
palanca grande (§6).

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

**RESUELTO (2026-09-11): la premisa de la compuerta se verificó primero, y pasa.** Decir "bit a
bit idéntico a `dc.load`" no significa nada si `dc.load` no es reproducible, así que eso se midió
antes que nada: dos corridas de `load_kndvi` en **procesos separados** sobre t18_600 (238 fechas)
dan valores, tiempos, coordenadas **y la coordenada `sensor`** idénticos. `sensor` registra de qué
producto vino cada adquisición, que es exactamente lo que se movería si el orden de empate fuera
inestable — así que la prueba toca el riesgo y no su alrededor. Procesos separados a propósito:
dentro de un mismo proceso un caché de índice tibio escondería justo la inestabilidad buscada.
**La identidad bit a bit sigue siendo una compuerta válida.** Harness:
`scripts/bench/check_load_determinism.py`.

**La prueba que queda, y es la única que hay que correr antes de decidir nada más:**

1. materializar **una** tesela (t18_600, que es la que tiene línea base en §6, §8.7 y §8.8);
2. correr `scripts/73` contra el Zarr y contra `dc.load`, mismos años;
3. diferencia de rásters con `np.array_equal(equal_nan=True)`, 10 bandas, manifiesto incluido
   — el mismo protocolo que validó §8.8;
4. `scripts/74` en ALL PASS.

Las otras salidas que la Fase 0 tenía que producir —fracción de NaN, tamaño comprimido, escalado
de lectura por hilos— **ya están medidas y están en §2**: 40,3 % NaN, 2,1x de compresión, ~2,2 TB
en total, y una descompresión que escala apenas 2,63x con 8 hilos pero que a un solo hilo ya son
~1,3 s contra ~700 s. Lo único que falta de la Fase 0 es el diff de rásters de arriba.

Sobre la alineación de chunks: se intentó medir la amplificación de lectura leyendo la misma
ventana de 333 px de un mosaico con chunks de 333 y de 256 px, y **el resultado salió al revés y
no es concluyente** (1,11 s alineado contra 0,24 s desalineado, con el store desalineado leyendo
*más* bytes). A esta escala la medición está dominada por costos fijos. No se persigue, porque la
pregunta se disuelve sola con el diseño ya elegido: **un store por tesela** significa que ninguna
tesela puede cruzar un chunk de otra, por construcción.

Si la igualdad bit a bit no se alcanza, la decisión que sigue es **declarar una tolerancia o
abandonar**, y esa es del autor, no del código.

**Notas de diseño, ya decididas.** El chunking va a lo largo de todo el eje temporal con
bloques espaciales modestos, porque el patrón de acceso es "toda la serie de tiempo de una
región espacial". Y **no** se guardan curvas pre-computadas en vez de kNDVI: hornean la
interpolación y la convención de ventana (D6) dentro del producto, y no ocupan menos.

Tres trampas más, todas de forma y no de tamaño:

- **Alineación de chunks, que es lo que decide si streamear de S3 sirve.** Si los chunks no
  caen sobre el retículo de teselas de D7, hay amplificación de lectura: una tesela de 333 px
  que cruza chunks de 256 px trae ~2x lo que usa. Chunkeando espacialmente contra la grilla de
  9.990 m / 333 px, cada tesela lee exactamente sus propios chunks. Si esto sale mal, streamear
  se ve pésimo por razones que no tienen nada que ver con S3.
- **Cantidad de objetos.** Chunks de ~10 MB para arriba. Algo como `(1, 256, 256)` produciría
  millones de objetos diminutos, que en Zarr v2 sobre S3 es doloroso de listar y leer (el
  sharding de Zarr v3 lo resuelve; más simple es no meterse). Time completo × tamaño de tesela
  da objetos de ~180 MB, uno por tesela.
- **Dispersión: un store por tesela, no un array gigante.** Sólo 5.769 de 17.019 teselas tienen
  vegetación nativa; un array denso sobre la extensión completa sería ~3x más grande con dos
  tercios vacío. Un Zarr por tesela evita la coordinación de escrituras por región, no
  desperdicia espacio, es trivialmente paralelo, y la reanudación es "¿existe el store de esta
  tesela?" — la misma lógica que ya usa `tile_progress.py`. Se pierde la elegancia del dataset
  único, pero calza con el patrón de acceso, porque la inferencia lee una tesela por vez.

**¿Se puede escribir un Zarr de este tamaño, y con qué máquina?** Sí, cómodamente, y es más
fácil de lo que suena porque **la escritura es en streaming y vergonzosamente paralela**: nunca
se sostiene el arreglo en memoria, cada worker escribe sus chunks y los olvida. El costo está
dominado por el lado de la *lectura*, que es la misma carga de COGs que ya se hace:

| | |
|---|---|
| por worker | ~732 MB de la tesela + ~2 GB de transitorio de carga ≈ **~3 GB** — igual que hoy, así que aplica la misma regla de 4 GB/vCPU de `docs/21` §8.9 |
| trabajo total | 5.769 × ~825 s ≈ **~1.322 horas-proceso** |
| en un m7i.16xlarge (64 vCPU) | ~21 h; dos nodos, ~10 h; en los 5 pods de Argo a `--jobs 6`, ~2 días |
| ancho de banda de escritura | 1 TB en 21 h ≈ **~14 MB/s agregados**. El PUT de S3 no es un problema |

**Y la jugada que probablemente conviene: no construirlo como un job aparte.** La próxima
corrida de producción ya carga cada tesela exactamente una vez. Un flag `--write-cube` en
`scripts/73` que escriba el Zarr como efecto secundario hace que construirlo no cueste
prácticamente nada más que el ancho de banda de escritura, entrega los mapas *y* el cubo en una
sola pasada, y deja todas las corridas siguientes en el camino rápido. Eso convierte un job de
21 horas en un flag. **Se decide después de la Fase 0**, no antes: si la igualdad bit a bit no
se alcanza, este atajo escribiría 1 TB de algo que no sirve.

## 4. La tensión de diseño: el cluster de dask estorba al procesamiento

La observación que ordena todo el escalado es que **las dos fases quieren máquinas
distintas**, y no un poco distintas:

| | construir la caché | inferir |
|---|---|---|
| límite | I/O, topado por el GIL a ~0,7 de núcleo (§8.1) | cómputo (forward torch) |
| forma que quiere | muchos procesos flacos, mucha CPU por GB | pocos procesos, o una GPU con vCPU al lado |
| paralelismo que paga | procesos, no hilos (6,2x contra 3,4x, §8.1) | hilos de torch hasta ~2 (89 % de eficiencia, §8.6) |
| dask | ayuda | **estorba** |

Esa última fila **está medida, no supuesta** (`docs/21` §8.10). Un `LocalCluster` de procesos
gana la carga con claridad —2,3x contra el scheduler de hilos— y aun así pierde la tesela: sus
workers siguen vivos durante la CNN y le suben el trabajo por año un 19 %, de modo que
proyectado a 27 años queda **peor que no hacer nada**. Es la misma patología de §6 y §8.2, por
la que el gateway recibe la tesela entera y no la carga, ahora con número.

Materializar el cubo **separa** las dos fases, y por eso permite por primera vez darle a cada
una la forma que quiere: la fase de construcción es justamente aquella donde el cluster de dask
sí gana, porque ahí no hay CNN detrás a la que gravar. Las tres opciones de abajo se diferencian
en *dónde* se pone ese corte.

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

### Lo que ya se descartó como restricción

**El índice ODC compartido no es el techo.** Era la objeción obvia a subir la concurrencia, y
se midió: satura, pero muy por encima de lo que este trabajo le pide — ~1 % del tiempo de una
tesela, ~0,23 q/s medios incluso a 128 procesos. El detalle está en `docs/21` §8.12. **No hace
falta sondearlo de nuevo antes de elegir opción**, y no limita el tamaño de nodo.

**El SSD local es una optimización, no un requisito.** Streamear el Zarr desde S3 alcanza: lo
que hace lenta la carga de hoy es el *número de requests y la latencia*, no los bytes —~5.970
aperturas de COG, 96 % costo fijo—, y el Zarr colapsa eso a unos pocos chunks por tesela, lo que
mueve el cuello de latencia a ancho de banda, que es donde S3 es bueno. En una sola pasada cada
tesela se lee exactamente una vez, así que bajar a SSD primero significa leer el TB **dos**
veces. El SSD paga cuando hay re-lecturas: re-corridas, reinicios, experimentación de
parámetros — que, dado el historial de reinicios de §8.5, puede muy bien ser el caso, pero es
cinturón y tiradores, no la base del diseño.

**Un beneficio adicional que sólo aparece con el cubo materializado:** deja de hacer falta
cargar el tramo completo por adelantado. Leer sólo la ventana de 3 años de cada objetivo se
vuelve barato, porque no hay penalidad de "reabrir los COGs" — y eso colapsa la huella de `obs`
que hoy manda en el dimensionamiento de RAM por vCPU (`docs/21` §8.9), lo que a su vez hace
mucho más fácil pasar a teselas de 20 km (`docs/21` §8.11).

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

**La regla de decisión, para no comprar por entusiasmo.** Hoy, con la carga en ~47 % de la
tesela, Amdahl topa la GPU en **~1,7x** por tesela: acelera lo que no es el cuello. La forma
útil de plantearlo es que una GPU no sólo hace rápida la CNN, sino que **libera a las CPU de
hacerla**, así que el mismo número de vCPU carga ~2,1x más teselas por hora. De ahí:

> Una instancia con GPU conviene sólo si cuesta menos que **~2x** una instancia CPU con el
> mismo número de vCPU.

A precios EC2 de hoy eso suele quedar cerca del empate, así que hay que **cotizarlo en el
momento de decidir**, no asumirlo. Lo que cambia el cálculo es exactamente lo de §2 punto 4:
con la carga materializada el forward pasa a ~77 % de la tesela y la GPU deja de ser 1,7x para
ser una palanca de 5x o más. Por eso el orden importa.

#### El problema no es sólo cuánto vale la GPU, es poder alimentarla

Esto es más fuerte que el argumento de Amdahl de arriba, y es la razón operativa por la que la
caché va primero.

En CPU, `--jobs N` compra el solapamiento gratis: N procesos hacen carga→CNN cada uno, así que
la espera de S3 de una tesela tapa la CNN de otra. **Eso no se traslada a una máquina con GPU**,
porque los N procesos quieren la misma GPU. Y `fan_out` reinvoca el script como N subprocesos
pasándole `--device` tal cual (`scripts/73_map_inference.py:628`), de modo que `--jobs 6
--device cuda` daría N contextos CUDA —cientos de MB de VRAM cada uno para un modelo de 14.535
parámetros— repartiéndose una GPU por time-slicing. Con un trabajo limitado por lanzamiento de
kernels, que es el nuestro, eso empeora justo lo que ya duele.

La forma correcta en una sola máquina es **cargadores como workers y la GPU en el driver**: dask
(o un pool de procesos) hace sólo la carga —la parte topada por el GIL y limitada por latencia
de S3, que sí escala con procesos— y un único proceso posee la GPU y consume teselas listas.
Notar que acá `load_client=True` deja de ser un problema y pasa a ser lo que se quiere: computar
al driver es correcto cuando el driver es el que tiene la GPU. Dos cuidados: **CUDA no sobrevive
a `fork`**, así que no hay que propagar `--device cuda` a los workers —si sólo cargan, no hay
CUDA en ningún worker y el problema no existe—; y cada cargador tiene que llamar a `dc.load`
**sincrónicamente** dentro del worker, no armar un grafo lazy que se reenvía al scheduler que lo
está corriendo, que es lo que costó 3.434 teselas (§8.5, §8.10).

**Y acá está el aguijón.** Los cargadores tienen que producir teselas al ritmo que la GPU las
consume. Un cargador son ~700 s de pared por tesela a ~0,72 de núcleo, así que:

| tiempo de GPU por tesela | cargadores concurrentes | vCPU sólo para cargar |
|---:|---:|---:|
| 120 s | ~6 | ~4 |
| 60 s | ~12 | ~8 |
| 20 s | ~35 | ~25 |

`t_gpu` no se sabe hasta que la prueba de arriba lo mida, y toda la tabla depende de él — otra
razón para correr esa prueba antes de dimensionar nada. Pero la conclusión no depende del valor
exacto: **sin el cubo materializado, un nodo con GPU tiene que cargar con suficiente vCPU para
correr entre 6 y 35 cargadores de Landsat concurrentes sólo para no dejar la GPU en hambre**, es
decir pagar precio de nodo GPU por una máquina que es sobre todo un cliente de S3.

Dicho al revés, y es la formulación que ordena todo este documento: **el cubo Zarr *es* la
separación entre carga y procesamiento**, hecha una sola vez, offline, en CPU barata, en vez de
rehacerla adentro de cada hora-GPU. La Opción A de §5 es esa misma separación hecha por corrida
en lugar de una sola vez.

Cuándo una GPU sí sería una decisión fácil para este tipo de procesamiento, para tenerlo de
referencia: un modelo materialmente más grande (un transformer, una U-Net sobre parches, o
muchos más miembros de ensemble), o trabajo por píxel pesado en vez de diminuto (ajuste denso
de series, kernels grandes, solvers iterativos). Nada de eso describe al modelo actual.

## 7. Secuencia

1. **Fase 0**, una tesela: consistencia bit a bit + fracción de NaN + tamaño comprimido +
   escalado de lectura por hilos + alineación de chunks. Compuerta: nada sigue si falla.
2. **Decidir `--write-cube`**: si la Fase 0 pasa, la próxima corrida de producción puede
   construir el cubo de paso y volver innecesaria la elección de (3). Es la decisión más barata
   de todo el plan y por eso va antes que la sonda.
3. **Sonda de capacidad**: ¿los ~5 workers son cuota o fueron circunstancia? Sólo hace falta si
   (2) sale que no.
4. **Elegir A, B o C** con (1) y (3) en la mano.
5. **Prueba de GPU** sobre una tesela cacheada, con tolerancia declarada y la regla de costo de
   §6. Independiente de (1)-(4); se puede correr en paralelo.
6. Construir.

## Referencias

- `docs/21` §8.1 (el GIL y el techo de 3,4x), §8.2 (por qué el gateway recibe la tesela
  entera), §8.4 (presupuesto), §8.5 (la corrida se detiene sola), §8.6 (1D-CNN y escalado de
  hilos de torch), §8.7 (tabla de smearing), §8.8 (memoria de la interpolación y
  `BIODIV_TILE_CACHE`), §8.9 (`--jobs 6`).
- `docs/21` §8.10 (por qué el cluster de dask gana la carga y pierde la tesela), §8.11 (el
  tamaño de tesela, que el cubo vuelve mucho más accesible), §8.12 (el índice ODC no es el
  techo).
- `scripts/argo/tile_progress.py` — el patrón de idempotencia contra S3 que las opciones A y C
  reutilizan.
