# Guía de Instalación y Uso — Sistema de Reconocimiento Facial

Guía paso a paso para configurar, ejecutar y operar el sistema de reconocimiento facial con Docker.

---

## Requisitos Previos

Antes de empezar, asegurate de tener instalado:

- **Docker Desktop** — [Descargar](https://www.docker.com/products/docker-desktop/)
- **Git** — Para clonar el repositorio
- Al menos **8 GB de RAM** disponibles para Docker
- **NVIDIA Container Toolkit** — Solo si vas a entrenar con GPU ([Guía de instalación](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

> **Nota:** No necesitás tener Python instalado localmente. Todo corre dentro de los contenedores Docker.

---

## 1. Clonar el Repositorio

```bash
git clone https://github.com/MatiiP/tuia-face-recognition-app.git
cd tuia-face-recognition-app
```

---

## 2. Levantar los Servicios con Docker

Desde la raíz del proyecto, ejecutá:

```bash
docker compose up --build -d
```

Este comando construye las imágenes Docker e inicia **4 servicios**:

| Servicio   | Puerto | URL                          | Función                          |
|-----------|--------|------------------------------|----------------------------------|
| backend   | 8000   | http://localhost:8000        | API REST (FastAPI)               |
| frontend  | 8080   | http://localhost:8080        | Interfaz gráfica (Gradio)        |
| postgres  | 5432   | —                            | Base de datos vectorial (pgvector)|
| jupyter   | 8888   | http://localhost:8888        | JupyterLab para experimentación  |

### Verificar que todo esté corriendo

```bash
docker compose ps
```

Deberías ver los 4 contenedores con estado `Up`.

### Ver logs del backend

```bash
docker compose logs -f backend
```

---

## 3. Abrir la Interfaz Gráfica (Gradio)

Abrí tu navegador e ingresá a:

**👉 http://localhost:8080**

La interfaz tiene 3 pestañas:

### Pestaña "Predecir"
1. Subí una imagen con uno o más rostros.
2. Hacé clic en **"Iniciar predicción"**.
3. Copiá el `job_id` que aparece.
4. Hacé clic en **"Consultar resultado de este job"** (o andá a la pestaña "Estado y resultados").
5. Se muestra la imagen con bounding boxes, keypoints y la identidad reconocida.

### Pestaña "Registrar identidad"
1. Escribí el nombre de la persona.
2. Subí una foto donde se vea **un solo rostro** claro.
3. Hacé clic en **"Registrar"**.
4. La identidad se guarda en la base de datos PostgreSQL.

### Pestaña "Estado y resultados"
1. Pegá un `job_id` de una predicción o registro anterior.
2. Hacé clic en **"Consultar"**.
3. Podés ver el JSON de resultados, la imagen anotada y los links de descarga.

---

## 4. Poblar la Base de Datos con las Identidades

Para cargar automáticamente 2 imágenes de cada persona (excepto Valentino) en la base de datos:

```bash
docker compose run --rm --entrypoint python jupyter seed_db.py
```

Deberías ver una salida similar a:

```
Registrando ambrogi — ambrogi_01.png... ✓
Registrando ambrogi — ambrogi_02.png... ✓
Registrando gianfranco — gianfranco_01.png... ✓
...
⏭ Saltando a valentino (excluido intencionalmente)

Seed finalizado: 12 identidades registradas con éxito.
```

### Verificar qué hay en la base de datos

```bash
docker compose exec postgres psql -U faces_user -d faces -c "SELECT name FROM embeddings;"
```

---

## 5. Entrenar (o Re-entrenar) el Modelo

### 5.1 Preparar el Dataset

Las imágenes deben estar organizadas en la carpeta `Dataset/` con la siguiente estructura:

```
Dataset/
├── ambrogi/
│   ├── ambrogi_01.png
│   ├── ambrogi_02.png
│   └── ...
├── gianfranco/
├── gianluca/
├── lucas/
├── matias/
├── roberto/
└── valentino/          ← Excluido del entrenamiento automáticamente
```

### 5.2 Ejecutar el Entrenamiento

#### Con GPU (Recomendado) ⚡

```bash
docker compose --profile gpu up jupyter-gpu --build
```

Luego abrí **http://localhost:8888/lab** y ejecutá todas las celdas de `train.ipynb`.

> **Nota:** La primera vez el build tarda ~15-20 min por la descarga de CUDA + PyTorch. Las siguientes veces usa cache y es casi instantáneo si no cambiaste `requirements.txt`.

#### Con CPU

```bash
docker compose up jupyter --build
```

Luego abrí **http://localhost:8888** y ejecutá `train.ipynb`.

### 5.3 Qué hace el notebook

El notebook ejecuta las siguientes etapas:

1. **Pipeline de 4 etapas:** Define las funciones de detección, alineación, extracción de embeddings y comparación.
2. **Carga de datasets:** Descarga LFW y carga el dataset local.
3. **Balanceo:** Aplica **undersampling** a LFW (cap 40 img/clase) y **oversampling** a clases locales (target 40 img/clase).
4. **Entrenamiento:** Fine-tuning de ResNet-50 durante 50 épocas con data augmentation agresiva.
5. **Métricas:** Genera curvas de loss/accuracy, matriz de confusión, AUC-ROC.
6. **Inferencia externa:** Evalúa el modelo sobre imágenes de la carpeta `Inferencia/`.

### 5.4 Parámetros configurables (en `train.ipynb`)

| Parámetro    | Valor por defecto | Descripción                              |
|-------------|-------------------|------------------------------------------|
| `EPOCHS`    | 50                | Número de épocas de entrenamiento        |
| `LEARNING_RATE` | 1e-4          | Learning rate (Adam optimizer)           |
| `TARGET_COUNT` | 40             | Target de imágenes por clase (balanceo)  |
| `FACE_SIZE` | 112               | Tamaño de rostro alineado (píxeles)      |
| `EMBEDDING_SIZE` | 512          | Dimensión del vector de embedding        |
| `SIMILARITY_THRESHOLD` | 0.75   | Umbral de similitud para inferencia      |

### 5.5 Después de entrenar

**Reiniciar el backend** para que cargue el modelo nuevo:

```bash
docker compose restart backend
```

**Limpiar la base de datos** (los embeddings viejos no son compatibles con un modelo nuevo):

```bash
docker compose exec postgres psql -U faces_user -d faces -c "DELETE FROM embeddings;"
```

**Re-poblar la base de datos** con el modelo nuevo:

```bash
docker compose run --rm --entrypoint python jupyter seed_db.py
```

---

## 6. Evaluación de Inferencia Externa

Después de entrenar, el notebook evalúa automáticamente el modelo sobre imágenes externas en `Inferencia/`:

```
Inferencia/
├── ambrogi/          # Persona conocida → debe reconocerla
├── gianfranco/       # Persona conocida → debe reconocerla
├── gianluca/         # Persona conocida → debe reconocerla
├── lucas/            # Persona conocida → debe reconocerla
├── matias/           # Persona conocida → debe reconocerla
├── roberto/          # Persona conocida → debe reconocerla
├── valentino/        # Desconocido → debe rechazarla
├── otros/            # Desconocidos → debe rechazarlas
└── varias_personas/  # Múltiples caras → debe detectar e identificar todas
```

Los resultados se guardan en `train_metrics/inference_report.json` y los gráficos en `Graphics/`.

---

## 7. Generar Gráficos de Evaluación (PCA / t-SNE)

Después de entrenar, podés generar visualizaciones de los embeddings:

```bash
docker compose run --rm --entrypoint python jupyter evaluate.py
```

Esto genera:
- `Graphics/pca.png` — Gráfico PCA de todos los embeddings (incluyendo Valentino).
- `Graphics/tsne.png` — Gráfico t-SNE con Valentino marcado con ★.
- `Graphics/evaluation_report.json` — Similitudes intra-clase y vecino más cercano de Valentino.

---

## 8. Consultar la Base de Datos

Abrir una consola SQL interactiva:

```bash
docker compose exec postgres psql -U faces_user -d faces
```

Consultas útiles:

```sql
-- Ver identidades registradas
SELECT name FROM embeddings;

-- Contar registros
SELECT count(*) FROM embeddings;

-- Ver estructura de la tabla
\d embeddings;

-- Salir
\q
```

---

## 9. Comandos Útiles

| Acción                              | Comando                                                    |
|-------------------------------------|-----------------------------------------------------------|
| Levantar todo                       | `docker compose up --build -d`                             |
| Levantar Jupyter con GPU            | `docker compose --profile gpu up jupyter-gpu --build`      |
| Levantar Jupyter con CPU            | `docker compose up jupyter --build`                        |
| Apagar todo                         | `docker compose down`                                      |
| Reiniciar backend                   | `docker compose restart backend`                           |
| Ver logs del backend                | `docker compose logs -f backend`                           |
| Evaluar modelo (PCA/t-SNE)          | `docker compose run --rm --entrypoint python jupyter evaluate.py` |
| Poblar base de datos                | `docker compose run --rm --entrypoint python jupyter seed_db.py` |
| Limpiar base de datos               | `docker compose exec postgres psql -U faces_user -d faces -c "DELETE FROM embeddings;"` |
| Abrir consola SQL                   | `docker compose exec postgres psql -U faces_user -d faces` |
| Ver contenedores activos            | `docker compose ps`                                        |

---

## 10. Estructura del Proyecto

```
tuia-face-recognition-app/
├── Dataset/                    # Imágenes organizadas por persona
│   ├── ambrogi/
│   ├── gianfranco/
│   ├── gianluca/
│   ├── lucas/
│   ├── matias/
│   ├── roberto/
│   ├── valentino/              # Excluido del entrenamiento
│   └── dataset.md              # Documentación del dataset
├── Inferencia/                 # Imágenes externas para test de inferencia
│   ├── ambrogi/                # Personas conocidas
│   ├── gianfranco/
│   ├── gianluca/
│   ├── lucas/
│   ├── matias/
│   ├── roberto/
│   ├── valentino/              # Persona desconocida
│   ├── otros/                  # Personas completamente desconocidas
│   └── varias_personas/        # Imágenes con múltiples rostros
├── models/
│   └── face_detection.pth      # Modelo entrenado (ResNet-50, 512-dim)
├── Graphics/                   # Todos los gráficos generados
├── train_metrics/              # Métricas de entrenamiento e inferencia
├── src/
│   ├── app/main.py             # Aplicación FastAPI
│   ├── lib/
│   │   ├── services/
│   │   │   └── face_service.py # Pipeline: Detección → Alineación → Embeddings → Comparación
│   │   └── schemas.py          # Modelos Pydantic
│   └── frontend/
│       └── gradio_ui.py        # Interfaz Gradio
├── train.ipynb                 # Notebook de entrenamiento (Pipeline completo)
├── evaluate.py                 # Script de evaluación (PCA/t-SNE)
├── seed_db.py                  # Script de carga de la base de datos
├── docker-compose.yml          # Definición de servicios Docker
├── Dockerfile                  # Imagen del backend (CPU)
├── Dockerfile.gpu              # Imagen del backend (GPU — NVIDIA CUDA)
├── Dockerfile.frontend         # Imagen del frontend
├── requirements.txt            # Dependencias Python
├── informe.md / informe.html   # Informe final del TP
└── guia_uso.md / guia_uso.html # Esta guía
```
