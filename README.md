# TP1 — Sistema de Reconocimiento Facial

**Materia:** Computación Visual — TUIA  
**Alumnos:** Gianfranco Frattini, Matias Prado

Sistema completo de reconocimiento facial basado en un pipeline de 4 etapas: **Detección → Alineación → Extracción de Embeddings → Identificación/Verificación**. Incluye una API REST asíncrona (FastAPI), persistencia con PostgreSQL + pgvector, y un frontend web.

---

## Estructura del Proyecto

```text
tuia-face-recognition-app/
├── src/                          # Código fuente de la aplicación
│   ├── app/main.py               # Punto de entrada FastAPI
│   ├── lib/
│   │   ├── api.py                # Rutas de la API
│   │   ├── config.py             # Configuración desde .env
│   │   ├── schemas.py            # Schemas Pydantic
│   │   ├── services/
│   │   │   ├── face_service.py   # Pipeline de reconocimiento
│   │   │   └── task_manager.py   # Manejo de tareas asíncronas
│   │   └── storage/
│   │       └── embedding_store.py # Persistencia (JSON / pgvector)
│   └── frontend/                 # Frontend Gradio
├── Dataset/                      # Imágenes por persona (custom)
├── Inferencia/                   # Imágenes externas para test de inferencia
├── models/                       # Modelo entrenado (.pth)
├── Graphics/                     # Gráficos de entrenamiento y evaluación
├── train_metrics/                # Métricas del entrenamiento (log + JSON)
├── output/                       # Resultados de inferencia del backend
├── data/                         # Embeddings persistidos
├── init-db/                      # Scripts de inicialización PostgreSQL
├── train.ipynb                   # Notebook de entrenamiento completo
├── evaluate.py                   # Script de evaluación (PCA, t-SNE)
├── seed_db.py                    # Carga inicial de embeddings a BD
├── Dockerfile                    # Imagen Docker (CPU)
├── Dockerfile.gpu                # Imagen Docker (GPU — NVIDIA CUDA)
├── Dockerfile.frontend           # Imagen Docker del frontend
├── docker-compose.yml            # Orquestación de servicios
├── requirements.txt              # Dependencias Python
└── README.md
```

---

## Requisitos Previos

- **Docker** y **Docker Compose** (v2+)
- **Python 3.12** (solo si se trabaja en entorno local sin Docker)
- **NVIDIA Container Toolkit** (solo para entrenamiento con GPU)

---

## Entrenamiento del Modelo

El proceso completo de entrenamiento está documentado en [`train.ipynb`](train.ipynb).

### Opción 1: Entrenar con Docker (GPU — NVIDIA) ⚡ Recomendado

Requiere [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado.

```bash
docker compose --profile gpu up jupyter-gpu --build
```

Luego abrí http://localhost:8888 y ejecutá `train.ipynb`.  
El notebook detecta automáticamente la GPU (`cuda` vs `cpu`).

### Opción 2: Entrenar con Docker (CPU)

```bash
docker compose up jupyter --build
```

Luego abrí http://localhost:8888 y ejecutá `train.ipynb`.

### Opción 3: Entrenar en entorno local

```bash
# Crear entorno virtual
uv venv --python 3.12 .venv
source .venv/bin/activate         # Linux/Mac
# .venv\Scripts\activate          # Windows

# Instalar dependencias
uv pip install -r requirements.txt

# Abrir Jupyter
jupyter lab
```

### Salidas del entrenamiento

El notebook genera automáticamente:

| Carpeta | Archivo | Descripción |
|---------|---------|-------------|
| `models/` | `face_detection.pth` | Modelo entrenado (mejor val accuracy) |
| `Graphics/` | `balanceo_antes_despues.png` | Comparación antes/después del balanceo |
| `Graphics/` | `distribucion_clases_train.png` | Balance de clases en el dataset |
| `Graphics/` | `loss_vs_epochs.png` | Curva de Loss (train + validation) |
| `Graphics/` | `accuracy_vs_epochs.png` | Curva de Accuracy (train + validation) |
| `Graphics/` | `confusion_matrix.png` | Matriz de confusión sobre validación |
| `Graphics/` | `auc_roc.png` | Curva AUC-ROC multiclase (One-vs-Rest) |
| `Graphics/` | `inferencia_similitudes.png` | Distribución de similitudes (inferencia) |
| `Graphics/` | `inferencia_accuracy.png` | Accuracy por persona (inferencia) |
| `train_metrics/` | `train_metrics.log` | Registro legible del entrenamiento |
| `train_metrics/` | `train_metrics.json` | Métricas estructuradas en JSON |
| `train_metrics/` | `inference_report.json` | Reporte de evaluación de inferencia |

---

## Levantar la Aplicación Completa

```bash
# Buildear y levantar todos los servicios
docker compose build
docker compose up -d
```

### Endpoints

| Servicio | URL |
|----------|-----|
| Backend (API) | http://localhost:8000 |
| Frontend | http://localhost:8080 |
| PostgreSQL | `localhost:5432` |
| Jupyter (CPU) | http://localhost:8888 |

### API REST

- `POST /insert` — Registrar una identidad (imagen + nombre)
- `POST /predict` — Ejecutar inferencia sobre imagen o video
- `GET /status/{job_id}` — Consultar estado de procesamiento asíncrono

---

## Pipeline de Reconocimiento Facial

### 1. Detección
Módulo de detección de **InsightFace** (`buffalo_s`) para extraer bounding boxes y 5 keypoints faciales.

### 2. Alineación
Transformación afín basada en los keypoints para normalizar la posición de ojos, nariz y boca a coordenadas estándar.

### 3. Extracción de Embeddings
Red **ResNet-50** (fine-tuned) que proyecta cada rostro alineado a un vector de **512 dimensiones**, normalizado L2.

### 4. Identificación / Verificación
Comparación por **similitud coseno** contra los embeddings almacenados. Umbral configurable (`SIMILARITY_THRESHOLD`).

---

## Modelo y Fine-Tuning

- **Arquitectura:** ResNet-50 pre-entrenada (ImageNet) + capa de embedding (512-d)
- **Hiperparámetros:** 50 épocas, Adam (lr=1e-4, weight_decay=1e-4), batch size 32
- **Data Augmentation:** Flip, rotación, affine, color jitter, blur, grayscale

---

## Dataset y Estrategia de Balanceo

### Fuentes de datos

- **LFW (Labeled Faces in the Wild):** Personalidades públicas con ≥70 imágenes (7 clases)
- **Custom:** Imágenes propias de los integrantes del equipo (6 clases)
- **Valentino:** Excluido del entrenamiento para validación de "desconocido"

### Balanceo: Undersampling + Oversampling

Para evitar sesgos, se aplica una estrategia de balanceo doble con un target de **40 imágenes por clase**:

- **Undersampling (LFW):** Las clases de LFW que superan las 40 imágenes se submuestrean aleatoriamente al target.
- **Oversampling (Locales):** Las clases locales que tienen menos de 40 imágenes se sobremuestrean (duplicación + data augmentation agresiva).

Esto asegura una distribución uniforme de ~40 imágenes por clase en el conjunto de entrenamiento, evitando que el modelo se sesgue hacia las clases con más datos.

---

## Evaluación de Inferencia Externa

La carpeta `Inferencia/` contiene imágenes completamente externas al dataset de entrenamiento, organizadas por persona:

```text
Inferencia/
├── ambrogi/          # Persona conocida
├── gianfranco/       # Persona conocida
├── gianluca/         # Persona conocida
├── lucas/            # Persona conocida
├── matias/           # Persona conocida
├── roberto/          # Persona conocida
├── valentino/        # Persona desconocida (excluida del entrenamiento)
├── otros/            # Personas completamente desconocidas
└── varias_personas/  # Imágenes con múltiples rostros
```

El notebook evalúa automáticamente:
- **Accuracy** para personas conocidas (si las identifica correctamente)
- **Tasa de rechazo** para desconocidos (si las marca como "desconocido")
- **Detección múltiple** para imágenes con varias personas
- Umbral de similitud: **0.75**

---

## Evaluación Visual (PCA / t-SNE)

El script `evaluate.py` genera:
- **PCA** y **t-SNE** de los embeddings faciales (incluyendo Valentino como "unknown")
- Reporte JSON con similitud intra-clase y vecino más cercano de Valentino

```bash
# Dentro del contenedor jupyter o en entorno local
python evaluate.py
```

---

## Configuración

Configurar mediante archivos `.env`:

1. **Docker:** Editar `.env.docker.example`
2. **Local:** Copiar `.env.local.example` a `src/.env`

Variables principales:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Nombre del archivo del modelo | `face_detection.pth` |
| `SIMILARITY_THRESHOLD` | Umbral de similitud coseno | `0.90` |
| `USE_PGVECTOR` | Usar PostgreSQL + pgvector | `true` |
| `EMBEDDING_DIM` | Dimensión del embedding | `512` |
| `FACE_SIZE` | Tamaño de rostro alineado | `112` |

---

## Notas Importantes

- El modelo entrenado final (necesario para el backend) puede descargarse desde este link:
  [Descargar face_detection.pth (Google Drive)](https://drive.google.com/file/d/1ZS7S05jn04c9TUVDIZfl9r0AqhiXgWle/view?usp=sharing)
  El archivo debe colocarse en la carpeta `models/` (es decir, `models/face_detection.pth`).
  **Nota:** Si ejecutás el proyecto usando Docker, el script `entrypoint.sh` intentará descargar automáticamente el modelo desde Google Drive si no encuentra el archivo en la carpeta `models/`.
- Para usar `pgvector`, levantar `postgres` y definir `USE_PGVECTOR=true` en `.env`
- El entrenamiento con GPU requiere NVIDIA Container Toolkit instalado en el host
- La carpeta `Inferencia/` no se incluye en la imagen Docker (se monta como volumen)
