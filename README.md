# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de detección, alineación, extracción de embeddings e identificación/verificación facial.

> 📄 El enunciado completo del trabajo práctico está en [ENUNCIADO.md](ENUNCIADO.md).

## Objetivo del backend

Implementar una API asincrónica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre una imagen (`/predict`)
- Consultar el estado del procesamiento asincrónico (`/status/{job_id}`)

La API responde `HTTP 202` con un `job_id` y luego permite consultar el resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tuia-face-recognition-app/
├── 1-pre-processing.ipynb       # Parte 1: dataset y preprocesamiento
├── 2-train.ipynb                # Parte 2: entrenamiento y comparación de modelos
├── src/
│   ├── app/
│   │   └── main.py              # App FastAPI + /health
│   ├── frontend/                # UI Gradio provista por la cátedra
│   │   ├── app.py
│   │   └── gradio_ui.py
│   └── lib/
│       ├── api.py               # Endpoints /upload, /insert, /predict, /status, /files
│       ├── config.py            # Configuración vía .env
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py  # ⚠️ Funciones a implementar
│       │   └── task_manager.py
│       └── storage/
│           ├── base.py
│           ├── embedding_store.py   # Persistencia en JSON
│           └── pgvector_store.py    # Persistencia en PostgreSQL + pgvector
├── models/                      # Modelo entrenado (.pth / .onnx), no se versiona
├── output/                      # Imágenes subidas y resultados, no se versiona
├── requirements.txt
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── .env.docker.example
└── .env.local.example
```

# Preparando el ambiente local

## Requisitos para trabajar de forma local

- Python 3.12
- Docker

## Configurá tu modelo

Entrená tu modelo y guardalo dentro de la carpeta `models`. El backend admite modelos con extensión **.pth** (PyTorch, se carga con `torch.load` el modelo completo) o **.onnx** (se carga con `onnxruntime`). Si usan otro framework, expórtenlo a **.onnx**.

Recordá actualizar la configuración del `.env` correspondiente (`MODEL_NAME`, `EMBEDDING_DIM`, `FACE_SIZE`, `SIMILARITY_THRESHOLD`) para que apunte a tu modelo.

> El backend no arranca si no existe el archivo `models/<MODEL_NAME>`.

Con docker compose, el entorno reinicia la aplicación y actualiza el código automáticamente al detectar cambios. Lo mismo ocurre en local si usan `--reload`.

El reinicio automático puede no funcionar en algunas versiones de Docker Desktop para *Windows*. En ese caso, vuelvan a correr los comandos del siguiente apartado para actualizar el código dentro de Docker.

## Opción 1 - Corriendo dentro de Docker

### 1. Construí y corré la aplicación

Ajustá el archivo **.env.docker.example** a tus necesidades. Luego corré desde la terminal:

```bash
docker compose build
docker compose up -d
```

> **Primer arranque:** el hook `post_start` de Postgres puede fallar la primera vez (`postgres hook exited with status 2`) porque el servidor todavía se está inicializando, y Compose se detiene antes de levantar el backend y el frontend. Es inofensivo (la aplicación crea la extensión `vector` por sí misma). Para evitarlo:
>
> ```bash
> docker compose up -d postgres   # puede mostrar ese error; ignorarlo
> docker compose ps               # esperar a que Postgres figure como (healthy)
> docker compose up -d            # levanta backend y frontend
> ```
>
> Luego verificar `http://localhost:8000/health`, `http://localhost:8000/docs` y `http://localhost:8080`. El backend tarda ~1 minuto en cargar los modelos. Requiere `det_10g.onnx` y `w600k_r50.onnx` en `models/` (ver [Modelo y fine-tuning](#modelo-y-fine-tuning)).

## Opción 2 - Configurando el ambiente local

### 1. Instalá uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Documentación de uv](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Configurá un ambiente virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activá el ambiente virtual

```bash
source .venv/bin/activate
```

### 4. Instalá las dependencias

```bash
uv pip install -r requirements.txt
```

### 5. Iniciá la base de datos

```bash
docker compose up postgres -d
```

### 6. Iniciá el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Iniciá el backend

Ajustá el archivo *.env.local.example* a tus necesidades. Desde la carpeta `src`, en otra terminal:

```bash
cp ../models/<NOMBRE_DE_TU_MODELO> models/
cp ../.env.local.example .env
uvicorn app.main:app --reload --port 8000
```

## Configuración

No hardcodeen parámetros: configúrenlos mediante `.env`.

1. Para la ejecución local, copien `.env.local.example` a `src/.env`.
2. Ajusten las variables del modelo, los paths y el threshold.
3. Configuren el detector: `DETECTOR_MODEL_NAME` (opcional, archivo `.pth`/`.onnx` en `models/`), `DET_SCORE_THRESHOLD`, `DET_INPUT_SIZE`, `MIN_FACE_SIZE` y `ALIGN_MARGIN`. Si su detector viene de una librería, sobrescriban `_load_detector` en `face_service.py`.
4. Opcional (habilitado por defecto): configuren la conexión a PostgreSQL + pgvector (`USE_PGVECTOR`). Con `USE_PGVECTOR=false`, los embeddings se guardan en un archivo JSON (`EMBEDDINGS_PATH`).

## Endpoints

- Backend: `http://localhost:8000` (documentación Swagger en `http://localhost:8000/docs`)
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (provisto por la cátedra): `http://localhost:8080`

## Pipeline

### Provisto por la cátedra

- API asincrónica (`/upload`, `/insert`, `/predict`, `/status`) y manejo de jobs.
- Carga del modelo (`.pth` / `.onnx`) desde `models/`.
- Búsqueda por similitud configurable (`cosine` o `l2`) e identificación (`identify`).
- Manejo de desconocidos con `SIMILARITY_THRESHOLD`.
- Persistencia configurable en JSON o en PostgreSQL + pgvector (`USE_PGVECTOR`).
- Frontend para registrar identidades, predecir y visualizar los resultados (cajas, etiquetas y keypoints).

### Implementado por el grupo (`src/lib/services/face_service.py`)

El pipeline es el mismo que se usó para preparar el dataset en `1-pre-processing.ipynb` y para evaluar en `2-train.ipynb`:

1. `detect_faces`: detector **SCRFD** (`det_10g.onnx` de InsightFace), cargado una sola vez en `_load_detector`. Descarta detecciones con score < `DET_SCORE_THRESHOLD` o con lado menor a `MIN_FACE_SIZE`. Devuelve `[]` si no hay rostros.
2. `align_face`: alineación por similitud con los 5 landmarks hacia la plantilla ArcFace (`insightface.utils.face_align.norm_crop`) a `FACE_SIZE` × `FACE_SIZE` (112×112, BGR `uint8`). Los landmarks salen de la detección sobre la imagen completa (idéntica a la del dataset) y se devuelven en coordenadas **relativas a la caja**. Si no se encuentran, se reintenta sobre un recorte con margen `ALIGN_MARGIN`.
3. `extract_embedding_from_face`: BGR → RGB, `(x - 127.5) / 127.5`, NCHW, inferencia con `onnxruntime` y normalización L2. Devuelve una `list[float]` de `EMBEDDING_DIM` (512).

La detección sobre la imagen completa se guarda por hilo, de modo que `predict` detecta una sola vez por imagen aunque tenga varios rostros. Los tests unitarios están en `tests/` (`PYTHONPATH=src pytest tests -v`, requieren los modelos en `models/`).

## Modelo y fine-tuning

**Modelo elegido:** InsightFace `buffalo_l` (ArcFace `w600k_r50.onnx` para reconocimiento y SCRFD `det_10g.onnx` para detección), sin fine-tuning.

**Descarga (link público de solo lectura):** <LINK_PUBLICO_MODELOS> — colocar `w600k_r50.onnx` y `det_10g.onnx` en `models/`.

| Item | Detalle |
|---|---|
| Arquitectura | ResNet-50 (IResNet) entrenada con la pérdida angular ArcFace; embedding de 512 dimensiones, normalizado L2 |
| Datos de preentrenamiento | WebFace600K (ver el repositorio de InsightFace) |
| Licencia | Pesos preentrenados de InsightFace: uso no comercial / investigación |
| Umbral | `SIMILARITY_THRESHOLD=0.357` (similitud coseno) |

### Comparación con una CNN propia

Se entrenó una CNN desde cero (4 bloques residuales con stride 2, PReLU y BatchNorm, global average pooling, embedding de 128 dimensiones L2; 4,9 M de parámetros) con Cross-Entropy sobre las 93 identidades de `train` (AdamW, lr 1e-3, weight decay 1e-4, coseno, batch 32, hasta 100 épocas con early stopping). Se probaron tres variantes y se eligió la de mayor accuracy de validación:

| Experimento | Val accuracy |
|---|---|
| Sin augmentation, dropout 0.4 | 74,7 % (train 100 %: memoriza) |
| **Con augmentation, sin dropout** | **87,9 %** |
| Con augmentation, dropout 0.4 | 74,7 % |

Ambos modelos se evaluaron con el mismo protocolo *open-set* sobre identidades que ninguno vio (`test_open`, 21 identidades):

| Métrica | CNN custom | InsightFace |
|---|---|---|
| AUC (verificación) | 0.921 | 1.000 |
| EER | 15.3 % | 0.08 % |
| TAR @ FAR=1e-3 | 16.9 % | 99.9 % |
| Top-1 (conocidos) | 74.7 % | 99.7 % |
| Precision / Recall / F1 (macro) | 0.584 / 0.695 / 0.565 | 0.987 / 0.996 / 0.991 |
| Rechazo correcto de desconocidos | 3.5 % | 98.3 % |
| Tiempo de inferencia CPU (ms/rostro) | 18.3 | 372.5 |
| Tamaño (MB) | 18.9 | 166.3 |

**Por qué se eligió InsightFace:** la CNN propia, entrenada con ~1.300 imágenes de 93 identidades, aprende a clasificar esas identidades pero sus embeddings no generalizan a personas nuevas (rechaza solo el 3.5 % de los desconocidos, lo que la vuelve inutilizable en la app). ArcFace, entrenado con millones de imágenes y un margen angular, separa casi perfectamente identidades no vistas. Se acepta su mayor latencia y su licencia no comercial.

**Cómo se calibró el umbral:** con el punto EER sobre pares de validación (`val_open`, identidades no usadas ni en entrenamiento ni en test). En `test_open` ese umbral da FAR = 0.03 % y FRR = 0.27 %.

**Análisis de errores:** los 2 falsos positivos de InsightFace (similitud 0.38–0.40) son personas con un parecido real (lentes y peinado similares); los falsos negativos corresponden a una misma foto con lentes de sol oscuros y uniforme comparada con fotos de frente. El detalle y las imágenes están en `2-train.ipynb`.

**Limitaciones:** `val_open` y `test_open` tienen ~20 identidades, y es posible que InsightFace haya visto estas figuras públicas durante su preentrenamiento, por lo que su desempeño con personas comunes puede ser menor.

## Dataset

- **Fuente:** LFW (*Labeled Faces in the Wild*), versión sin *funneled*. Uso con fines académicos y de investigación; las imágenes no se versionan en el repositorio.
- **Selección:** de 5.749 identidades y 13.233 imágenes se conservaron las identidades con al menos 10 imágenes (158 identidades, 4.324 imágenes), porque con 1 o 2 imágenes por persona una CNN desde cero no aprende nada útil (el 70,8 % de las identidades de LFW tiene una sola imagen).
- **Reglas de calidad** (en cascada, ver `1-pre-processing.ipynb`): archivos corruptos, sin rostro (6), varios rostros ambiguos (451), rostro menor a 40 px (0), borrosas por varianza del Laplaciano (114), exposición extrema (7) y casi-duplicados por dHash (42) y por similitud de embeddings > 0.95 (5); además se descartaron 3 probables errores de etiqueta. Se limitó a 50 imágenes por identidad para reducir el desbalance.
- **Resultado final:** 134 identidades y 2.748 imágenes (3.501 antes del tope de 50 por identidad, media de 20,6 imágenes por identidad).
- **Splits** (semilla 42, sin fuga de datos verificada):

| Split | Escenario | Identidades | Imágenes |
|---|---|---|---|
| train | closed | 93 | 1.290 |
| val | closed | 93 | 273 |
| test_closed | closed | 93 | 276 |
| val_open | open | 20 | 430 |
| test_open | open | 21 | 479 |

- **Preprocesamiento:** detección con SCRFD, alineación con 5 landmarks a 112×112 y *data augmentation* solo sobre `train` (flip horizontal, rotaciones y escalas suaves, brillo/contraste, ruido, blur, coarse dropout y compresión JPEG; se descartaron flips verticales, rotaciones fuertes y distorsiones elásticas tras medir cuánto alteran la identidad).
- **Sesgos y ética:** LFW está formado por figuras públicas de prensa, con predominio de hombres y de personas blancas, por lo que el desempeño puede ser menor en otros grupos. Los rostros son datos biométricos sensibles (Ley 25.326 de Protección de Datos Personales): no se suben imágenes de terceros al repositorio y las fotos usadas para probar la aplicación son de personas públicas, solo para demostración.

## Notas importantes

- Para usar `pgvector`, levanten `postgres` y definan `USE_PGVECTOR=true` en `.env`.
- Si cambian `EMBEDDING_DIM`, la tabla `embeddings` se recrea y se borran los registros previos.
- Los modelos (`*.pth`, `*.onnx`) y los datos no se versionan (ver `.gitignore`). Coloquen el modelo entrenado en `models/` y súbanlo a un link **público de solo lectura** para que los docentes puedan descargarlo.
