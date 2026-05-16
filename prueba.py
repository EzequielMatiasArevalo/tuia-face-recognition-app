import os
from dotenv import load_dotenv

# Cargar el archivo desde la ruta que acabas de crear
load_dotenv(dotenv_path="src/.env")

class Config:
    # Parámetros del modelo
    MODEL_NAME = os.getenv("MODEL_NAME", "VGG-Face")
    DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "opencv")
    THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", 0.4))
    
    # Paths
    DATASET_PATH = os.getenv("DATASET_PATH", "./data/dataset")
    
    # DB (Habilitada por defecto como pediste)
    DB_ENABLED = os.getenv("DB_ENABLED", "true").lower() == "true"
    DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

print(f"🚀 Configuración cargada: Modelo={Config.MODEL_NAME}, DB_Enabled={Config.DB_ENABLED}")