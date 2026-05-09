# Sistema de Reconocimiento Facial - Trabajo Práctico - Computer Vision

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)

## Miembros del Equipo
- Calabozo, Nicolas Daniel
- Lapolla, Martín Facundo

## Resumen del Proyecto y Stack Tecnológico

El presente proyecto abarca el diseño, entrenamiento y despliegue de un sistema completo de **Reconocimiento Facial**. El backend es una API asincrónica implementada en Python que permite registrar identidades, ejecutar inferencias sobre imágenes o videos, y consultar el estado de los procesamientos de forma fluida.

El **Stack Tecnológico** empleado en este trabajo práctico incluye:
- **Python 3.12** como lenguaje principal.
- **PyTorch** y **Facenet-PyTorch** para implementar las arquitecturas de red neuronal de visión artificial. Específicamente, se utilizó **MTCNN** para la detección y alineación facial, e **InceptionResnetV1** como modelo base para realizar un proceso de *Fine-Tuning* y la extracción de características (*embeddings*).
- **LFW (Labeled Faces in the Wild)** como el conjunto de datos base, junto con la librería **Albumentations** para la aplicación de *Data Augmentation* en un dataset propio enriquecido con imágenes adicionales de celebridades.
- **Scikit-Learn** para análisis de métricas (evaluación con K-Nearest Neighbors, Curvas ROC/AUC, Accuracy, y reducción de dimensionalidad con PCA y t-SNE).
- Base de datos **PostgreSQL** mediante la extensión **pgvector** para almacenar y persistir los *embeddings* extraídos y realizar búsquedas de similitud (midiendo distancias en el espacio vectorial).
- Entorno **Docker** y **Docker Compose** para la contenedorización y orquestación unificada del backend, la base de datos y la interfaz gráfica de usuario (Frontend).

## Preparando el ambiente local (Ejecución de Notebooks)

Para ejecutar localmente el proceso de entrenamiento y exploración de datos en los notebooks de Jupyter (`train.ipynb`), se recomienda encarecidamente utilizar [**uv**](https://docs.astral.sh/uv/) para gestionar el entorno virtual y la instalación de dependencias, garantizando una configuración rápida y minimizando conflictos de versiones.

### 1. Instalar uv

Para usuarios de Linux o macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Para usuarios de Windows:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configurar un entorno virtual con Python 3.12

```bash
uv venv --python 3.12 .venv
```

### 3. Activar el entorno virtual

En Linux/macOS:
```bash
source .venv/bin/activate
```
En Windows:
```bash
.venv\Scripts\activate
```

### 4. Instalar las dependencias del proyecto

```bash
uv pip install -r requirements.txt
```

De esta forma tendrás el ambiente correctamente configurado y listo para abrir el notebook mediante tu IDE favorito o ejecutando `jupyter notebook`.

## Corriendo la aplicación con Docker

Para levantar de manera íntegra toda la arquitectura de la aplicación (Frontend, Backend, y PostgreSQL con pgvector), el método recomendado y más directo es mediante Docker.

### 1. Configurar variables de entorno

Utilizá el archivo `.env.docker.example` de base para crear tu `.env`, ajustando las variables a tus necesidades y asegurándote de que los paths hacia tus modelos (*models*) sean los correctos.

### 2. Construir y ejecutar los contenedores

En la terminal, en la raíz del proyecto, ejecuta:

```bash
docker compose build
docker compose up -d
```

Una vez que los contenedores estén levantados y corriendo correctamente, podrás acceder a los siguientes servicios en tu navegador:
- **Backend API:** `http://localhost:8000`
- **Frontend UI:** `http://localhost:8080`
- **Base de Datos PostgreSQL/pgvector:** `localhost:5432`

*(Nota: El entorno local con o sin Docker actualiza el código automáticamente si utilizas Docker Compose. Si estás en Windows y el reinicio automático no funcionase, simplemente vuelve a construir la imagen de docker y a levantar los servicios).*
