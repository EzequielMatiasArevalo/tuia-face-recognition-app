# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de deteccion, alineacion, extraccion de embeddings e identificacion/verificacion facial.

## Objetivo del backend

Implementar una API asincronica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre imagen o video (`/predict`)
- Consultar estado de procesamiento asincronico (`/status/{job_id}`)

Tambien se exponen aliases alineados con la consigna:

- Registrar identidades (`/register`)
- Ejecutar inferencia sobre imagen o video (`/inference`)

La API responde `HTTP 202` con `job_id` y luego permite consultar resultado con estado:

```json
{
  "status": "done | inProgress | failed",
  "link": "url | none"
}
```

## Estructura

```text
tp1/
├── src/
│   ├── app/
│   │   └── main.py
│   └── lib/
│       ├── api.py
│       ├── config.py
│       ├── schemas.py
│       ├── services/
│       │   ├── face_service.py
│       │   └── task_manager.py
│       └── storage/
│           └── embedding_store.py
├── data/
│   └── embeddings.json
├── model/
├── output/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

# Preparando el ambiente local

## Requisitios para trabajar de forma local

- Python 3.12
- Docker

## Configura tu modelo

Entrena tu modelo y guardalo dentro de la carpeta models. Por defecto, el modulo soporta modelos construidos con pytorch validando la extension **.pth**.

Si eligen utilizar otro framework, pueden exportarlo a formato **.onnx**

Recuerda actulizar las configuraciones del .env correspondiente para actualizar la ruta hacia tu modelo.

El entorno local con o sin docker reinicia la aplicacion y actualiza el codigo automaticamente si uitlizan docker compose. 

Puede que el reinicio automatico no funcione en todas las versiones de Docker Desktop en sistemas *Windows*, en tal caso deberan correr los comandos como se mencionan en el siguiente apartado para actualizar el codigo dentro de docker.

## Opcion 1 - Corriendo dentro de docker

### 1. Buildea y corre la aplicacion.

Actualiza el archivo **.env.docker.example** y ajustalo a tus necesidades. Luego corre desde el terminal :

```bash
docker compose build
docker compose up -d
```

## Opcion 2 - Configurando el ambiente local

### 1. Install uv

Para usuarios de Linux o Mac:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows :

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

[Link](!https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) a la documentacion de uv.

### 2. Configura un ambiente virtual con python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activa el virtual environment

```bash
source .venv/bin/activate
```

### 4. Instala las dependencias.

```bash
uv pip install -r requirements.txt
```

### 5. Inicia la base de datos

```bash
docker compose up postgres -d
```

### 6. Incia el frontend

```bash
cd src
uvicorn frontend.app:app --port 8080
```

### 7. Inicia el backend

Asegurate de configurar el archivo *.env.local.example* para que se adapte a tus necesidades.

```bash
cp ../models/<YOUR MODEL NAME>.pth models
cp ../.env.local.example src/.env
uvicorn app.main:app --reload --port 8000 
```

## Configuracion

No hardcodear parametros. Configurar mediante `.env`:

1. En la ejecucion local copiar `.env.local.example` a `src/.env` dentro de la carpeta app
2. Ajustar variables de modelo, paths y threshold
3. Opcional ( habilitada por defecto ): configurar conexion a PostgreSQL + pgvector

## Endpoints

- Backend: `http://localhost:8000`
- PostgreSQL/pgvector: `localhost:5432`
- Frontend (imagen provista por catedra): `http://localhost:8080`

### Contrato async (paso 10)

- `POST /register` (alias: `/insert`) -> encola registro y responde `202` con `job_id`
- `POST /inference` (alias: `/predict`) -> encola prediccion y responde `202` con `job_id`
- `GET /status/{job_id}` -> devuelve `done | inProgress | failed`, `link`, `reason` y URLs publicas para artefactos cuando aplica

El flujo esperado es:
1. subir imagen (`/upload`)
2. encolar job (`/register` o `/inference`)
3. consultar `status` hasta `done`/`failed`

## Pipeline implementado (base funcional)

1. Deteccion de rostros con OpenCV Haar Cascade
2. Alineacion geometrica simple (recorte + normalizacion a `FACE_SIZE`)
3. Extraccion de embeddings (vector normalizado base)
4. Busqueda por similitud configurable (`cosine` o `l2`)
5. Manejo de desconocidos con `SIMILARITY_THRESHOLD`
6. Persistencia configurable en JSON o PostgreSQL + pgvector (`USE_PGVECTOR`)

## Modelo y fine-tuning

Completar en la entrega final:

- Arquitectura elegida (ResNet, EfficientNet, ViT, etc.)
- Justificacion tecnica y trade-offs
- Hiperparametros y proceso de fine-tuning
- Analisis de errores (FP/FN)
- Metricas: accuracy, precision, recall

## Dataset

Documentar:

- Fuente de imagenes publicas/provistas
- Cantidad por clase/persona
- Balance de clases
- Variaciones (iluminacion, pose, expresion)
- Reglas de filtrado/calidad

Para evitar datos personales, se recomienda usar LFW (`sklearn.datasets.fetch_lfw_people`) como dataset base.

## Evaluacion y defensa (paso 16)

Para facilitar evaluacion tecnica y demo, dejar evidencia en `train.ipynb` de:

- Metricas del sistema (accuracy, precision, recall)
- Curva ROC y criterio de threshold
- Analisis de falsos positivos/falsos negativos
- Visualizaciones de embeddings (PCA o t-SNE)
- Capturas/resultados de demo end-to-end con `job_id` y salida del backend

## CI (GitHub Actions)

En cada push/PR a `main` o `master`:

- `pytest` sobre `tests/` (contratos Pydantic + rutas API con `FaceService` mockeado; `docker compose config`; build de imagen `Dockerfile`.

Local:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest tests/ -q
docker compose config
```

## Notas importantes

- La implementacion actual es una base operativa para pruebas end-to-end y debe evolucionarse al modelo entrenado del equipo.
- Para usar `pgvector`, levantar `postgres` y definir `USE_PGVECTOR=true` en `.env`.
- Colocar el modelo entrenado en `models` el cual debera estar disponible en un link con acceso de solo lectura publico para poder ser descargado por los docentes.

