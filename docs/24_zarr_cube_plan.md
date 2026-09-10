# Cubo kNDVI materializado: plan de prueba y de escalado

**Estado (2026-09-10): la Fase 0 pasó, la prueba de GPU también, y el cubo ya se lee desde S3.** Lo que era un plan con
números derivados ya está medido: el cubo reproduce `dc.load` bit a bit y borra la carga (§3,
`docs/21` §8.14), y la GPU corre el forward 91x más rápido con los rásters dentro de la
tolerancia declarada (§6, `docs/21` §8.15). Las dos mediciones juntas cambian la arquitectura
elegida: **una pasada de construcción sola, después toda la inferencia contra el cubo en un nodo
G4 chico** — ver §3 y §7. El paso 6b —leer el cubo desde S3, que era la última incógnita
técnica— también está hecho y medido (`docs/21` §8.16): bit a bit idéntico, ~13 % de la tesela.
Donde un número siga siendo derivado y no medido, se dice.

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

**FASE 0 COMPLETA Y APROBADA (2026-09-11).** El diff de rásters está hecho: t18_600, años
2005/2015/2024, **3 rásters × 10 bandas todas bit a bit idénticas** contra la corrida de
`dc.load`, manifiesto idéntico en todas las columnas sustantivas, y `scripts/74` en ALL PASS.
La carga pasó de **763,8 s a 2,5 s (306x)** y la tesela de tres años de 867 s a 98 s. El detalle
y la extrapolación a 27 años están en `docs/21` §8.14; el código es `maptask.zarr_write` /
`_zarr_read` detrás de `BIODIV_TILE_ZARR`, con tests de ida y vuelta en `tests/test_maptask.py`.

**Entonces la compuerta ya no bloquea nada, y la decisión que sigue es §S2.3 (`--write-cube`).**
Lo que queda abajo es el registro de cómo se hizo la prueba.

**La prueba, tal como se corrió:**

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

**¿Se puede escribir un Zarr de este tamaño, y con qué máquina?** Sí, cómodamente. Pero el
motivo que daba este párrafo **no es el que aplica al código que existe**, y conviene decirlo
porque de ahí salía el dimensionamiento de RAM. Decía que "la escritura es en streaming y
vergonzosamente paralela: nunca se sostiene el arreglo en memoria, cada worker escribe sus
chunks y los olvida". **`load_tile` hace `.compute()`**, o sea que junta la tesela entera en el
driver y el driver escribe todos los chunks: al span de 27 años son ~820 MB residentes por
tesela concurrente, no cero. Es perfectamente pagable —la regla de 4 GB/vCPU de `docs/21` §8.9
lo cubre, y §8.10 midió el RSS del driver idéntico en todos los arms, así que `load_client=True`
no lo infla— y no se persigue la escritura distribuida, porque `zarr_write` esquiva
`xarray.to_zarr` a propósito (la codificación CF del tiempo rompería la identidad bit a bit,
§8.14) y porque escribir son 4-6 s contra ~300-690 s de carga. Lo que había que corregir es el
número, no la decisión. El costo está
dominado por el lado de la *lectura*, que es la misma carga de COGs que ya se hace:

| | |
|---|---|
| por worker | ~732 MB de la tesela + ~2 GB de transitorio de carga ≈ **~3 GB** — igual que hoy, así que aplica la misma regla de 4 GB/vCPU de `docs/21` §8.9 |
| en el driver | **~820 MB residentes por tesela concurrente** al span de 27 años — ver la corrección de abajo |
| trabajo total | 5.769 × ~825 s ≈ **~1.322 horas-proceso** |
| en un m7i.16xlarge (64 vCPU) | ~21 h; dos nodos, ~10 h; en los 5 pods de Argo a `--jobs 6`, ~2 días |
| ancho de banda de escritura | 1 TB en 21 h ≈ **~14 MB/s agregados**. El PUT de S3 no es un problema |

**Y la jugada que conviene, ahora medida: no construirlo como un job aparte.** La próxima
corrida de producción ya carga cada tesela exactamente una vez y ya produce mapas. Un flag
`--write-cube` en `scripts/73` que escriba el Zarr como efecto secundario entrega los mapas *y*
el cubo en una sola pasada. Lo que cuesta esa escritura se midió el 2026-09-11 sobre el arreglo
real de t18_600 (1.504 fechas, 667 MB crudos), contra los **763,8 s** que cuesta la carga de esa
misma tesela:

| zstd | escribir | MB | razón | % de la carga |
|---:|---:|---:|---:|---:|
| **1** | **4,2 s** | 335,8 | 1,99x | **0,6 %** |
| 3 | 7,4 s | 331,4 | 2,01x | 1,0 % |
| 5 | 13,3 s | 325,9 | 2,05x | 1,7 % |

**Nivel 1 es la elección**: 3,2x más rápido que el 5 y sólo 3 % más grande. Subir el nivel no
compra nada porque el arreglo no es muy comprimible de entrada (2x, §2), así que apretarlo más
es gastar CPU por gastarla.

**Eso parecía decidir la arquitectura, y la GPU lo dio vuelta (2026-09-10).** El argumento era:
construir el cubo como job aparte paga una fase de carga completa sólo para construirlo, mientras
que `--write-cube` cuesta ~0,6 % extra sobre una corrida que va a ocurrir igual. **Es correcto
mientras el forward corra en la misma CPU que hizo la carga.** Deja de serlo cuando el forward se
va a una GPU (§6, `docs/21` §8.15), porque entonces las dos fases quieren máquinas de precio muy
distinto — que es la misma tensión de §4, ahora con una diferencia de precio y no sólo de forma.

Por tesela, en segundos-núcleo de 27 años:

| | carga | forward | residuo | en qué nodo |
|---|---:|---:|---:|---|
| pasada de construcción | ~614 | — | — | spot de CPU, lo más barato que haya |
| pasada de inferencia (desde el Zarr) | ~2 | ~15 **s-GPU** | ~47 | un nodo G4 chico |
| *`--write-cube` empaquetado en un nodo con GPU* | ~614 | ~15 s-GPU | ~47 | **nodo GPU, 93 % de él esperando a S3** |

Empaquetar significa alquilar un nodo con GPU para que se quede ~10 minutos por tesela dentro de
la latencia de S3. Con un sobreprecio de ~1,35x eso sale **~20-25 % más caro** que separar, y no
compra nada: la GPU está ociosa durante casi todo. Y empaquetar sobre nodos de **CPU** es
bastante peor, porque ahí se pagan ~1.134 segundos-núcleo por tesela de forward que una GPU hace
en 15.

**Entonces: una pasada de construcción sola, y después toda la inferencia contra el cubo.** Que
es la **Opción A** de §5, la separación en dos etapas — no "sin objeto" después de todo, sino la
que gana, y por una razón que no estaba disponible cuando se escribió §5. Las Opciones B y C
siguen descartadas: B pierde el intermedio con el pod, y C depende de la sonda de capacidad.

La pasada de construcción, con lo medido: **5.769 × ~614 s ≈ ~984 horas-proceso** a
`--load-threads 4` (medido: 614 s por tesela contra 1.741 s en el camino síncrono, que es el
default de `TileConfig`; la diferencia importa y por eso `build_tile_zarr.py` expone el flag).
No necesita GPU, ni el modelo, ni los checkpoints — por eso puede correr en el pool más barato
que haya. Es idempotente por tesela, así que spot calza y una interrupción cuesta una tesela.
`--write-cube` en `scripts/73` sigue siendo útil como red de seguridad si alguna corrida de
producción sale antes que el cubo, pero ya no es la jugada principal.

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

**El SSD local es una optimización, no un requisito — la conclusión aguanta, el argumento no.**
Streamear el Zarr desde S3 alcanza, y eso ahora está medido (`docs/21` §8.16): lo que hace lenta
la carga de hoy es el *número de requests y la latencia*, no los bytes —~5.970 aperturas de COG,
96 % costo fijo—, y el Zarr colapsa eso a unos pocos chunks por tesela, lo que mueve el cuello de
latencia a ancho de banda, que es donde S3 es bueno.

Lo que este documento decía para descartarlo era que **"bajar a SSD primero significa leer el TB
dos veces", y es falso**: los bytes cruzan la red exactamente una vez en los dos casos, y lo que
la copia agrega es un viaje por disco, no una segunda lectura de S3. La distinción que sí manda
es otra: **si la transferencia se solapa o no con el cómputo.** Sin solapar, copiar pierde (4,5 s
contra 3,1 s de leer directo). Solapando, el disco local gana fuerte —leer de ahí son 0,49 s
contra 3,1 s— y la copia de ~4 s se esconde entera dentro de los ~70 s de GPU de la tesela
anterior, lo que baja el costo visible de ~13 % a ~1 % de la tesela. Y hay lugar de sobra: a
`--jobs 4` el enlace necesita ~22 MB/s contra los ~150 MB/s medidos en agregado.

Así que el SSD del nodo GPU vale **~12 % del tiempo de tesela y sólo con un hilo de prefetch**.
Sigue siendo cinturón y tiradores y no la base del diseño, pero por una razón distinta de la que
estaba escrita: no porque leer dos veces salga caro, sino porque leer directo ya sale barato.

**Un beneficio adicional que sólo aparece con el cubo materializado:** deja de hacer falta
cargar el tramo completo por adelantado. Leer sólo la ventana de 3 años de cada objetivo se
vuelve barato, porque no hay penalidad de "reabrir los COGs" — y eso colapsa la huella de `obs`
que hoy manda en el dimensionamiento de RAM por vCPU (`docs/21` §8.9), lo que a su vez hace
mucho más fácil pasar a teselas de 20 km (`docs/21` §8.11).

### Cómo elegir

No hace falta decidir ahora, y no conviene: la Fase 0 produce el número que separa A de B
—cuánto pesa una tesela comprimida y cuánto tarda en subir y bajar de S3—, y la sonda de
capacidad separa C de las otras dos. El orden es Fase 0 → sonda de capacidad → elección.

## 6. La GPU: medida, y es la palanca grande (2026-09-10)

**RESUELTO. La prueba se corrió y el detalle con todas las tablas está en `docs/21` §8.15**; acá
queda lo que decide el diseño. Nodo **Tesla T4 (15 GB), 8 vCPU**, cuatro teselas materializadas
como Zarr primero, de modo que lo único que separa las dos ramas es el dispositivo.

| | resultado |
|---|---|
| forward solo, 46.553 px, 5 semillas | 51,4 s (CPU 1 hilo) → **0,56 s**, **91x** |
| tesela de 3 años desde el Zarr, `--torch-threads 1` | 141,2 s → **11,0 s** |
| año-tesela en régimen | ~0,90 ms/px → ~44 µs/px, **~21x** |
| tesela de 27 años, extrapolada en el mismo nodo | ~1.148 s → **~62 s** |
| compuerta de rásters, tolerancia declarada de antemano | **PASA**, peor 5,85e-6 del rango de la banda contra 1e-4 permitido |

**Lo que este documento predecía sobre el *cómo* era falso, y en la dirección cómoda.** No está
limitado por lanzamiento de kernels: el tamaño de batch no mueve nada y capturarlo como CUDA
graph no compra nada, así que **no hay que subir el batch ni agrupar entre teselas**. Agrupar las
cinco semillas —la arm C de §8.13— pierde también en GPU, y peor que en CPU (0,24x). PCIe es ~5 %.

**Y el miedo de "no se puede alimentar" tampoco se sostuvo.** `--jobs 4 --device cuda` corre
cuatro procesos sobre una sola T4 **sin degradarse** (10,5–12,7 s por tesela contra 11,0 s en
solitario), con ~640 MiB de VRAM por contexto: entrarían ~24 en la tarjeta. Pasar a 8 procesos
casi duplica el tiempo por tesela y compra sólo +19 %, así que la rodilla está entre **4 y 6
procesos por T4**. No hace falta el diseño de "cargadores como workers y la GPU en el driver" que
se proponía más abajo en esta sección: con el cubo materializado no hay nada que cargar, y
`--jobs N --device cuda` simplemente funciona.

**La regla de compra, ahora con números.** El costo por tesela es el precio por vCPU-hora por los
segundos-núcleo que el nodo necesita, así que la GPU conviene mientras `p_gpu/p_cpu` sea menor
que la razón de segundos-núcleo:

| pasada | CPU | GPU | razón de equilibrio |
|---|---:|---:|---:|
| **desde el Zarr** (toda corrida posterior) | ~1.148 s | ~62 s | **~18x** |
| **primera pasada, todavía leyendo COGs** | ~1.762 s | ~676 s | **~2,6x** |

Las formas G4 de una sola GPU salen ~1,3–1,4x un M7i equivalente por vCPU —**cotizar al momento
de decidir, no asumir**— así que conviene en las dos, pero de manera abrumadora sólo en la
primera fila. Eso es lo que da vuelta a §3, y está discutido ahí.

**Qué máquina.** Como cada proceso quiere ~1 vCPU para el residuo y la rodilla está en 4–6
procesos por GPU, lo que calza es **una GPU chica con 4–8 vCPU** (`g4dn.xlarge`/`2xlarge`). Un
nodo grande de una GPU deja vCPU ociosas; uno multi-GPU está sobre-equipado por un factor de
~4. Y **G5/G6e no compran nada**: 14.535 parámetros y ~640 MiB de VRAM no llenan una tarjeta
grande, y el término que la GPU ataca ya bajó a 55 s por tesela de 27 años.

Para la corrida completa desde el cubo: **~99 horas-núcleo y ~24 horas-GPU** contra ~1.840
horas-núcleo en CPU. La inferencia deja de ser un problema de flota y pasa a ser un nodo G4 chico
corriendo alrededor de un día.

Cuándo una GPU sería una decisión fácil para este tipo de procesamiento, para tenerlo de
referencia: un modelo materialmente más grande (un transformer, una U-Net sobre parches, o muchos
más miembros de ensemble), o trabajo por píxel pesado en vez de diminuto. Nada de eso describe al
modelo actual — y sin embargo la GPU gana igual, porque el forward es ~95 % de lo que queda una
vez que la carga está materializada.

## 7. Secuencia

1. ~~**Fase 0**, una tesela: consistencia bit a bit + fracción de NaN + tamaño comprimido +
   escalado de lectura por hilos + alineación de chunks.~~ **HECHO y aprobado** (`docs/21` §8.14).
2. ~~**Decidir `--write-cube`**~~ **HECHO, y la respuesta cambió**: la construcción va como pasada
   propia y la inferencia contra el cubo, porque separar es ~20-25 % más barato que empaquetar una
   vez que el forward está en una GPU (§3). `--write-cube` queda como red de seguridad.
3. ~~**Sonda de capacidad**~~ **sin objeto para esta ruta**: la inferencia ya no necesita una flota
   —son ~24 horas-GPU y ~99 horas-núcleo— y la construcción es Argo, que es lo que ya se sabe
   operar. Volvería a hacer falta sólo si se quisiera la Opción C para acortar la construcción.
4. ~~**Elegir A, B o C**~~ **HECHO: Opción A**, la separación en dos etapas (§3, §5).
5. ~~**Prueba de GPU**~~ **HECHA y pasada** (§6, `docs/21` §8.15): 91x sobre el forward, ~21x sobre
   el año-tesela, 4 procesos por T4 sin degradarse, rásters dentro de 5,85e-6 del rango de banda
   contra 1e-4 declarado.
6. **Construir.** Lo que queda, en orden:
   - **6a. La pasada de construcción**: un job derivado de
     `scripts/bench/build_tile_zarr.py` sobre las 5.769 teselas, escribiendo un store por tesela
     a S3, **`--workers 7`**, clevel 1, spot de CPU barato, reanudación por "¿existe el store?".
     ~2,3 TB, ~50 USD al mes de almacenamiento.

     **El scheduler ya no es `--load-threads 4`, y eso cambia el costo a la mitad y algo más**
     (`docs/21` §8.17). Este documento venía costeando la construcción con la forma que §8.10
     recomienda para `scripts/73`, que es la afinada alrededor de la CNN — una restricción que
     la construcción no tiene, como decía §4 sin haberlo medido. Medido ahora sobre una tesela
     completa: **688,8 s con `threads=4` contra 292,2 s con `cluster=7`, 2,36x**, y los rásters
     bit a bit idénticos entre las dos ramas y contra el store de Fase 0. Las **~984
     horas-proceso** de este plan pasan a **~417 horas-nodo** si el 2,36x se sostiene al span de
     27 años.

     Salvedad honesta: el cluster gana la pared pagando ~1,45x de segundos-núcleo (~701 contra
     ~482 por tesela). Por nodo alquilado —que es como se paga— gana igual. Lo que no se midió,
     y cerraría la pregunta, es correr **N construcciones `threads=4` concurrentes** en un mismo
     nodo: la carga está limitada por latencia y no por CPU, así que caben varias.
   - ~~**6b. Leer el cubo desde S3**~~ **HECHO y medido** (`docs/21` §8.16): `zarr_write` y
     `_zarr_read` entienden `s3://`, y las cuatro teselas leídas desde un prefijo real salen
     **bit a bit idénticas** a los stores locales. Directo de S3 son ~3,1 s por tesela en
     solitario y ~8,3 s a `--jobs 4`, o sea **~13 % de la tesela** al span de 27 años contra los
     ~70 s de GPU de §8.15. Dos cosas cambiaron de paso: cortar el eje temporal es gratis en
     tamaño y mejor en las dos puntas, así que pasa a ser el default (128 fechas por chunk); y
     como S3 no tiene rename, la escritura atómica se apoya en escribir los atributos al final,
     de modo que un store a medias lee como *miss* y no como tesela corta.
   - **6c. La corrida de inferencia** en un `g4dn.xlarge`/`2xlarge` con `--jobs 4`.

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
