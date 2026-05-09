# Informe Final: Sistema de Reconocimiento Facial (TP1)

**Materia:** Computación Visual — TUIA  
**Alumnos:** Gianfranco Frattini, Matias Prado

## 1. Arquitectura General y Flujo de Trabajo

El sistema está construido como una aplicación de microservicios desplegada en **Docker**, compuesta por:

| Servicio     | Tecnología           | Puerto | Función                                      |
|-------------|----------------------|--------|----------------------------------------------|
| `backend`   | FastAPI + PyTorch    | 8000   | API REST: detección, registro y predicción   |
| `frontend`  | Gradio               | 8080   | Interfaz gráfica de usuario                  |
| `postgres`  | PostgreSQL + pgvector| 5432   | Base de datos vectorial de embeddings        |
| `jupyter`   | JupyterLab           | 8888   | Entorno de experimentación                   |
| `jupyter-gpu` | JupyterLab + CUDA | 8888   | Entrenamiento acelerado con GPU NVIDIA       |

### Pipeline de Reconocimiento (face_service.py)

El pipeline sigue 4 etapas claramente separadas dentro del código:

1. **Detección** (`detect_faces`): Utiliza **RetinaFace** (vía InsightFace `buffalo_s`) para encontrar bounding boxes y 5 keypoints faciales (ojos, nariz, comisuras de la boca).
2. **Alineación** (`align_face`): Aplica una **transformación afín** (`norm_crop`) usando los keypoints para rotar, escalar y centrar cada rostro a un tamaño estándar de 112×112 píxeles.
3. **Embeddings** (`extract_embedding_from_face`): El rostro alineado pasa por un modelo convolucional (ResNet-50 fine-tuned) que produce un **vector de 512 dimensiones**, normalizado con L2.
4. **Comparación** (`identify`): Los embeddings de la consulta se comparan contra la base de datos PostgreSQL usando **Similitud Coseno**. Si el score supera el umbral configurado, se asigna la identidad; caso contrario, se marca como `unknown`.

## 2. Dataset

### Fuentes de datos

Se construyó un dataset combinando dos fuentes:

| Fuente | Clases | Imágenes/persona | Descripción |
|--------|--------|-------------------|-------------|
| **LFW** (Labeled Faces in the Wild) | 7 | 70–530 (original) | Personalidades públicas con ≥70 imágenes |
| **Custom** (propio) | 6 | 4–15 | Imágenes de los integrantes del equipo |

### Personas del dataset

| Persona     | Fuente | Imágenes originales | Rol                                     |
|-------------|--------|--------------------|-----------------------------------------|
| Ambrogi     | Custom | 4                  | Prueba de clase minoritaria             |
| Gianfranco  | Custom | 15                 | Integrante del equipo                   |
| Gianluca    | Custom | 14                 | Integrante (hermano de Valentino)       |
| Lucas       | Custom | 11                 | Integrante del equipo                   |
| Matías      | Custom | 12                 | Integrante del equipo                   |
| Roberto     | Custom | 10                 | Integrante del equipo                   |
| **Valentino** | Custom | **Excluido** | **Prueba fuera de distribución (unknown)** |
| 7 clases LFW | LFW  | 70–530 c/u         | Presidentes y personalidades públicas   |

**Valentino** es hermano genético de Gianluca. Al excluirlo del entrenamiento, podemos evaluar empíricamente si el modelo capta rasgos familiares compartidos o lo clasifica como desconocido.

### Estrategia de Balanceo: Undersampling + Oversampling

El dataset original presentaba un fuerte desbalance: las clases LFW tenían hasta 530 imágenes, mientras que las locales apenas 4–15. Para resolverlo se implementó una **estrategia de balanceo doble** con un target de **40 imágenes por clase**:

- **Undersampling de LFW:** Cada clase de LFW que supera las 40 imágenes en el split de train se submuestrea aleatoriamente al target. Esto evita que el modelo se sesgue hacia las caras de presidentes.
- **Oversampling de clases locales:** Las clases con menos de 40 imágenes se sobremuestrean duplicando imágenes existentes. La **Data Augmentation agresiva** (rotación, flip, color jitter, blur, grayscale, affine) asegura que las copias no sean idénticas.

Resultado: distribución uniforme de ~40 imágenes por clase en el conjunto de entrenamiento.

## 3. Entrenamiento (Fine-Tuning)

### Configuración

| Parámetro       | Valor            |
|-----------------|------------------|
| Backbone        | ResNet-50 (timm, pre-trained ImageNet) |
| Embedding size  | 512              |
| Épocas          | 50               |
| Batch size      | 32               |
| Learning rate   | 1e-4 (Adam, weight_decay=1e-4) |
| Loss            | CrossEntropyLoss |
| Data augmentation | HorizontalFlip, Rotation(15°), Affine, ColorJitter, GaussianBlur, RandomGrayscale |
| Dispositivo     | GPU (NVIDIA CUDA) / CPU |

### Docker GPU

El entrenamiento se realizó utilizando Docker con soporte para GPU NVIDIA, lo que permite una aceleración significativa del proceso:

```bash
docker compose --profile gpu up jupyter-gpu --build
```

El `Dockerfile.gpu` utiliza la imagen base `nvidia/cuda:12.4.1-runtime-ubuntu22.04` con Python 3.12 y todas las dependencias del proyecto. El notebook detecta automáticamente la disponibilidad de CUDA.

### Exportación del Modelo

El modelo final se exporta como `nn.Sequential(backbone, embedding_head)`, lo que permite cargarlo directamente en la API sin dependencias del script de entrenamiento. Se guarda automáticamente el mejor modelo según val accuracy en `models/face_detection.pth`.

## 4. Evaluación

### Métricas de Entrenamiento

El notebook genera automáticamente:
- **Curvas de Loss y Accuracy** (train vs validation) por época
- **Matriz de Confusión** sobre el conjunto de validación
- **Curva AUC-ROC multiclase** (One-vs-Rest) con micro-average

### Prueba de Valentino (Unknown)

Como test de generalización, se compara el embedding de Valentino (excluido del entrenamiento) contra Gianluca (su hermano). Esto evalúa si el modelo distingue personas genéticamente similares o produce falsos positivos.

### Evaluación de Inferencia Externa

Se implementó una sección completa de evaluación sobre imágenes **externas al dataset** de entrenamiento, almacenadas en la carpeta `Inferencia/`:

| Subcarpeta | Tipo | Evaluación |
|-----------|------|------------|
| `ambrogi/`, `gianfranco/`, `gianluca/`, `lucas/`, `matias/`, `roberto/` | Personas conocidas | Accuracy de reconocimiento |
| `valentino/`, `otros/` | Personas desconocidas | Tasa de rechazo correcto |
| `varias_personas/` | Imágenes grupales | Detección e identificación de múltiples caras |

**Umbral de similitud:** 0.75. Si la similitud máxima contra el banco de referencia no supera este umbral, la persona se clasifica como "desconocido".

**Métricas generadas:**
- Accuracy por persona conocida
- Tasa de rechazo para desconocidos (falsos positivos vs rechazos correctos)
- Distribución de similitudes (histograma conocidos vs desconocidos)
- Reporte completo en `train_metrics/inference_report.json`

### Evaluación Visual (PCA / t-SNE)

El script `evaluate.py` genera visualizaciones de reducción de dimensionalidad:
- **PCA** — Proyección a 2 componentes principales
- **t-SNE** — Proyección no lineal con Valentino marcado con ★
- Reporte JSON con similitud intra-clase y vecino más cercano de Valentino

## 5. Base de Datos (Seed)

Se pre-cargaron **2 imágenes por persona** (excepto Valentino) en PostgreSQL mediante el script `seed_db.py`, totalizando **12 registros** en la tabla `embeddings`.

## 6. Archivos Generados

| Archivo                                | Descripción                                       |
|----------------------------------------|---------------------------------------------------|
| `models/face_detection.pth`            | Modelo PyTorch entrenado (ResNet-50, 512-dim)     |
| `Graphics/balanceo_antes_despues.png`  | Comparación antes/después del balanceo            |
| `Graphics/distribucion_clases_train.png` | Distribución final de clases (train)            |
| `Graphics/loss_vs_epochs.png`          | Curva de Loss (train + validation)                |
| `Graphics/accuracy_vs_epochs.png`      | Curva de Accuracy (train + validation)            |
| `Graphics/confusion_matrix.png`        | Matriz de confusión sobre validación              |
| `Graphics/auc_roc.png`                 | Curva AUC-ROC multiclase                          |
| `Graphics/inferencia_similitudes.png`  | Distribución de similitudes (inferencia)          |
| `Graphics/inferencia_accuracy.png`     | Accuracy por persona (inferencia)                 |
| `Graphics/pca.png`                     | Gráfico PCA de embeddings                         |
| `Graphics/tsne.png`                    | Gráfico t-SNE de embeddings                       |
| `train_metrics/train_metrics.log`      | Registro legible del entrenamiento                |
| `train_metrics/train_metrics.json`     | Métricas estructuradas en JSON                    |
| `train_metrics/inference_report.json`  | Reporte de evaluación de inferencia               |
| `Graphics/evaluation_report.json`      | Similitud intra-clase + vecinos de Valentino      |
| `train.ipynb`                          | Notebook de entrenamiento (Pipeline completo)     |
| `evaluate.py`                          | Script de evaluación (PCA/t-SNE/métricas)         |
| `seed_db.py`                           | Script de carga inicial de la base de datos       |
