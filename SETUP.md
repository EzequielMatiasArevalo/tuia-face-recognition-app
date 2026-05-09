# Cómo correr el sistema

Guía paso a paso, de cero hasta tener todo corriendo y el notebook validado.

## Requisitos

- **Docker Desktop** (Windows, Mac o Linux).
- **Git**.
- **Python 3.12** instalado en el host (solo para correr el script de batch register y el notebook).
- ~1 GB de espacio en disco.

No hace falta instalar Python deps a mano para el sistema en sí — todo corre adentro de Docker.

## 1. Clonar y preparar el `.env`

```bash
git clone <url-del-repo>
cd tuia-face-recognition-app
cp .env.docker.example .env
```

El archivo `.env.docker.example` ya viene con todos los valores comentados. **Para el TP no hace falta cambiar nada** — los valores por defecto funcionan.

Si querés ajustar algo común:

| Variable | Para qué |
|---|---|
| `SIMILARITY_THRESHOLD` | Más alto = más estricto. Default `0.55`. |
| `USE_PGVECTOR` | `true` = Postgres, `false` = JSON local (necesario para que el notebook lea los datos). |
| `MODEL_NAME` | Pack de InsightFace. `buffalo_l` (default, mejor) o `buffalo_s` (más liviano). |

Ver el `.env.docker.example` para la lista completa con explicaciones.

## 2. Levantar el sistema

```bash
docker compose build
docker compose up -d
```

La primera vez tarda varios minutos (descarga el pack `buffalo_l` ~300 MB).

Verificar que está todo arriba:

| Servicio | URL | Cómo verificar |
|---|---|---|
| Backend | http://localhost:8000/health | `{"status":"ok","model":"buffalo_l"}` |
| Frontend | http://localhost:8080 | Se abre la UI con tres tabs |

## 3. Cargar el dataset

Hay dos formas: a mano por la UI o automática con un script.

### Opción A — Script automático (recomendado)

1. Organizar las fotos así:

```
dataset/
  alejandro/
    foto1.png
    foto2.png
    ...
  roberto/
    ...
  desconocido/        # fotos de gente NO registrada (para test "fuera del dataset")
    ...
```

2. **Asegurarse que `USE_PGVECTOR=false`** en el `.env` (necesario para que el notebook lea los datos):

```powershell
(Get-Content .env) -replace '^USE_PGVECTOR=.*', 'USE_PGVECTOR=false' | Set-Content .env
docker compose restart backend
```

3. Vaciar el JSON de embeddings (si tenía registros viejos):

```powershell
Set-Content -Path data\embeddings.json -Value '[]'
```

4. Correr el script (no necesita instalar nada, usa solo stdlib):

```bash
python scripts/batch_register.py dataset/
```

El script:
- Divide automáticamente las fotos de cada persona en 70% train / 30% test.
- Registra el set de train (POST `/insert`).
- Predice el set de test y verifica que la identifique correctamente.
- Si existe `dataset/desconocido/`, predice cada foto y verifica que devuelva `unknown`.
- Imprime un reporte con accuracy.

### Opción B — A mano por la UI Gradio

1. Abrir http://localhost:8080.
2. Tab **Registrar identidad**: subir foto, escribir nombre, click **Registrar**.
3. Tab **Predecir**: subir foto, click **Iniciar predicción** y luego **Consultar resultado**.

Útil para probar fotos sueltas, no práctico para registrar muchas.

## 4. Correr el notebook de validación

El notebook lee los embeddings registrados, calcula métricas y genera todos los gráficos (PCA, t-SNE, FP/FN, threshold sweep, matriz de confusión, métricas por persona, conclusiones).

### Opción A — Headless (línea de comandos)

```powershell
# Crear venv minimo si no existe (una sola vez)
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe jupyter nbconvert ipykernel numpy matplotlib scikit-learn

# Ejecutar el notebook in place (los outputs quedan guardados en el .ipynb)
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace train.ipynb
```

### Opción B — VS Code

1. Abrir `train.ipynb`.
2. Click en **Select Kernel** → elegir `.venv` (o el venv que tengas con jupyter instalado).
3. **Run All**.
4. **Ctrl+S** para guardar (los outputs quedan adentro del `.ipynb`).

## 5. Detener / reiniciar

| Comando | Para qué |
|---|---|
| `docker compose stop` | Apaga los containers (datos persisten) |
| `docker compose down` | Apaga y elimina los containers (datos persisten en el volumen) |
| `docker compose down -v` | Apaga, elimina containers **y borra la base Postgres** |
| `docker compose restart backend` | Reinicia solo el backend (necesario tras cambiar `.env`) |
| `docker compose up -d --build` | Reconstruye y arranca (después de cambios en código) |

## Problemas comunes

| Síntoma | Causa | Solución |
|---|---|---|
| `docker compose build` falla con "permission denied" | Docker Desktop no está corriendo | Abrir Docker Desktop |
| `/health` devuelve 500 con "model path does not exist" | InsightFace todavía está descargando el pack | Esperar 1–2 min y reintentar |
| El notebook dice "No hay embeddings cargados todavía" | `USE_PGVECTOR=true` (van a Postgres) o no se registró nada | Cambiar a `false` y volver a registrar |
| `pip install` falla en Windows con error de C++ | Falta VC++ Build Tools | Usar Docker (no instalar local) |
| Puerto 8000 / 8080 / 5432 ya en uso | Otra app en ese puerto | Cerrar la otra app o cambiar el puerto en `docker-compose.yml` |
| El script `batch_register.py` falla con "module not found" | Estás en un Python distinto al esperado | El script usa solo stdlib, debería funcionar con cualquier `python` 3.x |
| `[FAIL] foto.png  ValueError: Exactly one face must be detected` | La foto tiene más de una cara | Recortar la foto para que solo se vea una cara, o saltearla |

## Resumen rápido (TL;DR)

```bash
# 1. Setup
cp .env.docker.example .env
docker compose build
docker compose up -d

# 2. Cargar dataset
# (poner fotos en dataset/<persona>/)
python scripts/batch_register.py dataset/

# 3. Correr notebook
.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace train.ipynb
```
