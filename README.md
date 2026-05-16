# TP1 - Sistema de Reconocimiento Facial

Plantilla base para desarrollar un sistema completo de deteccion, alineacion, extraccion de embeddings e identificacion/verificacion facial.

## Objetivo del backend

Implementar una API asincronica en Python que permita:

- Registrar identidades (`/insert`)
- Ejecutar inferencia sobre imagen o video (`/predict`)
- Consultar estado de procesamiento asincronico (`/status/{job_id}`)

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
│   ├── processed/          # Dataset LFW filtrado y estructurado
│   ├── aligned_faces/      # Rostros recortados y normalizados 160x160
│   └── embeddings.json     # Base de datos vectorial en formato JSON
├── models/
│   ├── facenet_finetuned.pth           # Modelo final entrenado mediante Transfer Learning
│   └── inceptionresnetv1_finetuned.pth # Copia de face_detection.pth
├── output/                 # Almacenamiento JSON con los resultados de predict
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

## Pipeline implementado (base funcional)

1. Deteccion de rostros con OpenCV Haar Cascade
2. Alineacion geometrica simple (recorte + normalizacion a `FACE_SIZE`)
3. Extraccion de embeddings (vector normalizado base)
4. Busqueda por similitud configurable (`cosine` o `l2`)
5. Manejo de desconocidos con `SIMILARITY_THRESHOLD`
6. Persistencia configurable en JSON o PostgreSQL + pgvector (`USE_PGVECTOR`)

## Dataset

Documentar:

- Fuente de imagenes (propias + publicas/provistas)
- Cantidad por clase/persona
- Balance de clases
- Variaciones (iluminacion, pose, expresion)
- Reglas de filtrado/calidad


## Modelo y fine-tuning

Completar en la entrega final:

- Arquitectura elegida (ResNet, EfficientNet, ViT, etc.)
- Justificacion tecnica y trade-offs
- Hiperparametros y proceso de fine-tuning
- Analisis de errores (FP/FN)
- Metricas: accuracy, precision, recall



Justificación técnica

Se eligió FaceNet + InceptionResnetV1 porque tiene:

Alto rendimiento en tareas de reconocimiento facial
Buen equilibrio entre precisión y costo computacional
Disponibilidad de pesos preentrenados robustos (vggface2)
Compatibilidad directa con PyTorch
Facilidad para realizar fine-tuning parcial



* Motivos de elección

- Detección facial

MTCNN permite:
Detectar múltiples rostros
Obtener bounding boxes precisas
Localizar keypoints faciales
Mejor robustez frente a pose e iluminación


- Alineación facial
Luego de detectar el rostro:

Se recorta la región facial
Se normaliza a resolución fija 160x160
Se realiza conversión RGB
Se aplica normalización:
(face_tensor - 127.5) / 128.0
La alineación garantiza consistencia geométrica antes de generar embeddings.



- Extracción de embeddings:

Cada rostro alineado es procesado por FaceNet para obtener:
Vector embedding de dimensión 512
Embedding normalizado por norma L2
Esto permite comparar rostros mediante similitud coseno.


Comparación por similitud
Se implementaron dos métricas:
cosine similarity
L2 similarity


Trade-offs:
	
Ventaja                                         Desventaja 

Muy buena precisión facial                    	Modelo relativamente pesado
Embeddings robustos	                            Mayor consumo de memoria
Fine-tuning sencillo	                          Inferencia más lenta que modelos lightweight
Compatible con GPU y CPU	                      Requiere imágenes alineadas correctamente


Pipeline de Procesamiento:

Cada vez que el backend recibe una imagen en /predict, hace los siguientes 6 pasos de forma automática:

Detección de Rostros: Usa la librería MTCNN para escanear la foto completa, encontrar dónde hay caras y marcar sus coordenadas. Si hay una placa de video (GPU/CUDA) disponible la usa para ir más rápido, y si no, usa el procesador (CPU).

Recorte y Alineación: Agarra cada cara detectada, la recorta de la foto original y ajusta su tamaño para que quede en un cuadrado perfecto de 160x160 píxeles, que es el tamaño que exige el modelo.

Extracción de Características (Embeddings): El rostro recortado pasa por la red neuronal, la cual analiza los rasgos y los traduce en una lista de 512 números (un vector).

Normalización: El vector se achica para que su longitud total (norma L2) valga exactamente 1. Esto permite comparar un vector con otro más adelante.

Comparación y Búsqueda: El sistema compara esos 512 números contra todos los rostros que ya tenías guardados. Usa la métrica de Similitud Coseno para ver qué tan parecidos son (valores entre 0 y 1).

Filtro de Desconocidos: Para evitar que confunda a un extraño con alguien registrado, se usa un umbral (SIMILARITY_THRESHOLD = 0.6). Si el parecido es menor a 0.6, el sistema decide que la persona es "unknown" (desconocida).



El Modelo Base

Se utilizó la red neuronal InceptionResnetV1, que ya viene preentrenada de fábrica con millones de rostros (dataset VGGFace2). Esta red es excelente porque combina la velocidad de ResNet con la precisión de los bloques Inception.


Fine tuning(Transfer Learning)

Para lograr que el modelo reconozca a las personas específicas, se hizo un entrenamiento dirigido:

Se congelaron los bloques internos del modelo para no arruinar lo que ya sabía sobre detectar ojos, cejas, narices y sombras.

Se activaron para el entrenamiento las últimas capas de la red (last_linear y last_bn) y se le agregó una capa de salida nueva (clasificador) adaptada a la cantidad de personas de tu dataset.

Se usó el optimizador Adam configurado con dos velocidades (Learning Rates) diferentes: muy lento para las capas internas (para ajustar detalles finos sin romper nada) y más rápido para la capa de salida (para que aprenda rápido a separar las identidades).



Resultados de las Pruebas

El rendimiento del modelo modificado se evaluó usando el dataset estándar LFW (Labeled Faces in the Wild) con una separación del 30% de las imágenes para test. Los resultados fueron los siguientes:

Precision: 99.80% (De todo lo que el modelo dijo que era "Persona X", el 99% era correcto).

Recall:    99.67% (De todas las fotos reales de "Persona X", el modelo logró encontrar el 99,67).

F1-score:  99.73%

Accuracy LFW: 99.86% (El porcentaje de aciertos totales sobre el dataset de pruebas).



Análisis de los Resultados (PCA y t-SNE)

Para comprobar visualmente si el entrenamiento funcionó, se agarraron los vectores de 512 números y se redujeron a gráficos de 2 dimensiones:

PCA: Mostró que las líneas principales de variación en los datos corresponden a las diferencias entre personas.

t-SNE: Este gráfico demostró de forma contundente que las fotos de la misma persona se agrupan en islas o nubes súper compactas y separadas de los demás. Esto confirma que el modelo separa muy bien a los individuos y que va a tener muy pocos errores en producción.


Dataset Utilizado

El sistema se entrenó combinando fotos propias con imágenes del dataset público LFW.
Para asegurar la calidad del entrenamiento, se filtró el dataset para quedarse únicamente con aquellas personas que tuvieran al menos 40 fotos disponibles (seleccionando las 20 identidades con más imágenes). El set incluye muchas dificultades reales, como cambios bruscos de luz, caras de perfil, anteojos, expresiones de alegría o seriedad, y diferentes resoluciones.



----------------------------

docker compose build
docker compose up -d

Documentación de la API (Swagger): http://localhost:8000/docs
Interfaz Gráfica (Gradio): http://localhost:8080

-----------------------------

powershell -ExecutionPolicy ByPass -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

uv venv --python 3.12 .venv
.venv\Scripts\Activate.ps1

uv pip install -r requirements.txt

docker compose up postgres -d

cd src
$env:BACKEND_URL="http://localhost:8000"
uvicorn frontend.app:app --port 8080

## Notas importantes

- La implementacion actual es una base operativa para pruebas end-to-end y debe evolucionarse al modelo entrenado del equipo.
- Para usar `pgvector`, levantar `postgres` y definir `USE_PGVECTOR=true` en `.env`.
- Colocar el modelo entrenado en `models` el cual debera estar disponible en un link con acceso de solo lectura publico para poder ser descargado por los docentes.

