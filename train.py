import os
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import fetch_lfw_people
from facenet_pytorch import MTCNN, InceptionResnetV1
from tqdm import tqdm
from dotenv import load_dotenv

# Importar las herramientas locales de tu proyecto
import sys
# Calculamos la ruta absoluta a la carpeta 'src' basándonos en la ubicación de este archivo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))
from src.lib.storage.pgvector_store import PgVectorEmbeddingStore
from src.lib.schemas import EmbeddingRecord

# 1. CONFIGURACIÓN INICIAL
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")
# Usamos la ruta relativa al proyecto para encontrar el .env
# train.py está en la raíz, así que la raíz es simplemente dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Cambia esto a '.env' si ya creaste tu archivo propio, o deja el de ejemplo para probar
load_dotenv(os.path.join(PROJECT_ROOT, '.env.docker.example')) 

# 2. FUNCIÓN PARA PREPARAR EL MODELO
def get_model(num_classes):
    # CRÍTICO: classify=True le dice a la red que pase por la capa 'logits'
    model = InceptionResnetV1(pretrained='vggface2', classify=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    # Reemplazamos el head para el entrenamiento
    model.logits = nn.Linear(512, num_classes).to(device)
    for param in model.logits.parameters():
        param.requires_grad = True
        
    return model

# 3. DATASET PARA sklearn LFW
class SklearnLFWDataset(Dataset):
    def __init__(self, images, labels, mtcnn):
        self.images = images
        self.labels = labels
        self.mtcnn = mtcnn

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Sklearn devuelve floats entre 0 y 1. MTCNN espera numpy uint8 o PIL
        img_np = (self.images[idx] * 255).astype(np.uint8)
        
        # MTCNN devuelve tensor (3, 160, 160)
        face = self.mtcnn(img_np)
        if face is None:
            face = torch.zeros(3, 160, 160)
            
        return face, self.labels[idx]

# 4. CARGA DE DATOS SKLEARN
def prepare_data(batch_size=32):
    print("Descargando/Cargando LFW de sklearn...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=1.0, color=True)
    
    mtcnn = MTCNN(image_size=160, margin=20, device=device, post_process=True)
    dataset = SklearnLFWDataset(lfw.images, lfw.target, mtcnn)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, lfw, mtcnn

# 5. LOOP DE ENTRENAMIENTO
def train_head(model, loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.logits.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        epoch_loss = 0
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
            
    print("Entrenamiento finalizado.")
    return model

# 6. GUARDAR EMBEDDINGS EN LA BASE DE DATOS VECTORIAL
def generate_and_store_embeddings(lfw_data, mtcnn, store: PgVectorEmbeddingStore):
    print("Generando embeddings y guardando en PGVector...")
    
    # Cargamos el backbone SIN la capa de clasificación (classify=False) para sacar embeddings
    embedding_model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
    embedding_model.eval()

    loop = tqdm(range(len(lfw_data.images)))
    for idx in loop:
        img_np = (lfw_data.images[idx] * 255).astype(np.uint8)
        label_id = lfw_data.target[idx]
        person_name = lfw_data.target_names[label_id]
        
        # Detectar
        face = mtcnn(img_np)
        if face is None:
            continue
            
        # Generar embedding (agregamos dimensión de batch)
        face = face.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = embedding_model(face).cpu().numpy().flatten()
            
        # Guardar en BD usando los schemas de tu proyecto
        record = EmbeddingRecord(
            id_imagen=str(uuid.uuid4()),
            embedding=embedding.tolist(),
            path=f"lfw_virtual_path_{idx}.jpg", # Como viene de sklearn no tenemos el path real
            etiqueta=person_name,
            metadata={"source": "LFW Sklearn"}
        )
        store.append(record)

# 7. EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # 1. Preparar datos
    train_loader, lfw_data, mtcnn = prepare_data(batch_size=32)
    
    # 2. Configurar rutas de guardado
    model_dir = os.getenv("MODEL_PATH", "models")
    model_name = os.getenv("MODEL_NAME", "inception_resnet_lfw_finetuned.pth")
    save_path = os.path.join(model_dir, model_name)
    
    # 3. Entrenar o Cargar el modelo
    model = get_model(len(lfw_data.target_names))
    
    if os.path.exists(save_path):
        print(f"El modelo ya existe en {save_path}. Cargando pesos guardados (saltando entrenamiento)...")
        model.load_state_dict(torch.load(save_path, map_location=device))
    else:
        print("Iniciando el entrenamiento del modelo...")
        trained_model = train_head(model, train_loader, epochs=5)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(trained_model.state_dict(), save_path)
        print(f"Modelo guardado exitosamente en: {save_path}")
    
    # 4. Inicializar base de datos vectorial (Psycopg 3)
    # Aquí asegúrate de tener tu Docker corriendo (docker compose up -d)
    try:
        # El .env dice 'postgres', pero eso solo funciona DENTRO de Docker. 
        # Como ejecutas desde Windows, el host debe ser 'localhost'.
        db_host = os.getenv("POSTGRES_HOST")
        if db_host == "postgres":
            db_host = "localhost"

        store = PgVectorEmbeddingStore(
            host=db_host,
            port=int(os.getenv("POSTGRES_PORT", 5432)),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", 512)) # Convertimos a int por si acaso
        )
        
        # 5. Generar embeddings y guardar en DB
        generate_and_store_embeddings(lfw_data, mtcnn, store)
        print("¡Proceso completo! Caras guardadas en la base de datos.")
    except Exception as e:
        print(f"Error conectando a la base de datos (revisa tu Docker): {e}")
