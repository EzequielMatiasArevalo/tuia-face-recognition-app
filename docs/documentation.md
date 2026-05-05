# Documentación del modelo, TP1 - Reconocimiento facial

## 1. Arquitectura del modelo

Se utilizó ResNet18 con pesos pre-entrenados en ImageNet (1.28M imágenes) con el fin de no tener que entrenar el modelo desde cero, reduciendo los requerimientos de datos propios. La arquitectura produce embeddings de 512 dimensiones, suficientes para detectar diferencias sutiles en rostros.

Como alternativas se consideraron: ResNet50, que ofrece mayor capacidad pero tiene un costo computacional más alto. Luego, EfficientNet tiene inferencia más lenta para embeddings, mientras que ViT requiere más datos y cómputo.

## 2. Proceso de fine-tuning

### Dataset

Se combinaron el dataset LFW (1.288 imágenes, 7 personas públicas) con un dataset propio (60 imágenes, 5 personas: Matías, Agustina, Isabel, Vicky y Guillermo), resultando en un total de 1.348 imágenes de 12 clases. El split utilizado fue 80/20 estratificado por clase.

### Etapa 1: Feature extraction (5 épocas)
El backbone se congeló y solo se entrenó la cabeza de clasificación con lr=1e-3 y batch size 8. Accuracy resultante: train 77.5% / val 77.8%.

### Etapa 2: Fine-tuning completo (10 épocas)
Se descongeló el backbone usando learning rates diferenciadas para no degradar los pesos pre-entrenados: 1e-6 para el backbone y 1e-4 para la cabeza.

Accuracy final: train 95.6% / val **90.2%**, una mejora de 12.4 pp sobre la etapa anterior. Loss: CrossEntropyLoss. Resolución de entrada: 224x224.

## 3. Métricas de evaluación
Sobre el validation set, el modelo alcanzó accuracy 90.2%, precision 89.7%, recall 88.3% y F1 89.0%.

## 4. Análisis de errores
Los falsos positivos se deben principalmente a similitud de pose e iluminación entre personas distintas, también por parecido físico, por ejemplo, entre padre e hijo. Se mitigan con el umbral de similitud coseno a 0.55. Los falsos negativos ocurren en variaciones extremas de pose, y el data augmentation durante el entrenamiento reduce este efecto.

Los pares con mayor confusión fueron Matías/Guillermo en el dataset propio y George W. Bush/George H. W. Bush en el subconjunto LFW.

## 5. Arquitectura del sistema

El modelo recibe una imagen y devuelve un vector de 512 dimensiones sin conocimiento de identidades ni del conjunto de entrenamiento. La lógica de comparación toma ese vector, lo compara contra los embeddings registrados en PostgreSQL con pgvector mediante similitud coseno, y determina si supera el umbral de 0.55 configurado en el '.env'. La capa de presentación traduce ese resultado en una identidad o en "unknown".

Esto garantiza que una imagen sin registro previo siempre devuelva "unknown", incluso si estuvo en el conjunto de entrenamiento, ya que el modelo no clasifica directamente sino que compara contra lo registrado.

## 6. Descarga del modelo

Modelo: `resnet18_facial_recognition_full.pth`, 44.8 MB. 

[Link de descarga](https://drive.google.com/file/d/1WF1IszgtigB9BkQhJfumf8yakFHL7OCX/view?usp=drive_link).