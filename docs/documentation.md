# Documentación del modelo, TP1 - Reconocimiento facial

## 1. Arquitectura del modelo

Para el reconocimiento uso InsightFace, con el paquete buffalo_l. El backbone que hace el trabajo pesado es ArcFace (w600k_r50, basado en ResNet50), que ya viene entrenado sobre datasets enormes de caras. Lo que devuelve son embeddings de 512 dimensiones, normalizados, que alcanzan de sobra para distinguir rostros distintos.

El mismo paquete buffalo_l trae también el detector (det_10g, RetinaFace) y el modelo de landmarks (1k3d68, 2d106det), que se usan en las etapas de detección y alineación.

Descarté entrenar algo propio o hacer fine-tuning de un backbone, principalmente porque mi dataset es chico: 127 imágenes en total, 5 personas. Con esa cantidad, entrenar desde cero es directamente inviable, y hacer fine-tuning también es riesgoso porque el modelo se sobreajusta rápido. Un modelo ya entrenado en millones de rostros trae representaciones de identidad facial que se pueden aprovechar directamente sin entrenar nada más.

Las opciones que evalué:

- Entrenar desde cero: necesitaría cientos de miles de imágenes para converger a algo útil. Con 127 no hay forma.
- Fine-tuning de un backbone genérico (ResNet18/50 pre-entrenado en ImageNet): mejor que lo anterior, pero ImageNet no está pensado para caras, y con pocas clases y pocas imágenes por clase el overfitting llega rápido.
- Modelo pre-entrenado específico para caras (ArcFace/InsightFace): la que elegí. No hay que entrenar nada y las representaciones ya están optimizadas justo para este problema.

## 2. Fine-tuning

No entrené ni ajusté el backbone. Utilicé ArcFace tal cual, por transferencia de aprendizaje pura: ya está resuelto para el problema de identificación facial en general, entrenado sobre datasets con millones de identidades.

Las razones:

- El dataset es chico (127 imágenes, 5 clases) y hacer fine-tuning ahí es casi garantía de overfitting.
- No hacía falta: el modelo pre-entrenado ya separa perfecto las identidades del dataset (accuracy 1.00 en test), así que no había margen de mejora real que justificara tocar los pesos.
- El costo de entrenar un backbone de este tamaño pide GPU y tiempo que no tenía disponible (no dispongo de GPU dedicada), sin garantías de que mejore algo con tan pocos datos.

## 3. Dataset

Armé un dataset propio de 127 imágenes con 5 personas, familiares (Agustina, Guillermo, Isabel, Matías, Vicky), entre 25 y 26 fotos por persona para garantizar dataset balanceados. Corresponden a imágenes que varían en pose, luz y expresión, dentro de lo que se puede lograr con celular en casa con distintas calidades de cámara.

También probé sumar LFW (fetch_lfw_people, 1288 imágenes, 7 personas públicas), pero terminé descartándolo. Las imágenes de LFW vienen en 125x94 px con un crop muy cerrado que no llega a mostrar el mentón, y RetinaFace necesita ver la cara completa para poder detectarla bien. Probé agrandar las imágenes (resize a 400px con LANCZOS) y bajar el umbral de detección (det_thresh=0.2), pero no sirvió , el problema no era de resolución ni de confianza del detector, era que directamente faltaba estructura facial en el crop. Así que me quedé solo con el dataset propio.

Durante el registro, si una imagen no tiene exactamente una cara detectada (0 o 2+), se descarta.

Para el split, separé 5 imágenes por persona para test (25 en total) en una carpeta aparte (dataset_tp1_test/), distinta de la carpeta de registro (dataset_tp1/, con las 102 restantes). Que la separación sea física y no solo lógica en memoria evita que se cuele contaminación entre train y test si vuelvo a correr el notebook.

## 4. Métricas

Resultados sobre el test set (25 imágenes, 5 por persona):

| Persona    | Precision | Recall | F1   |
|------------|-----------|--------|------|
| Agustina   | 1.00      | 1.00   | 1.00 |
| Guillermo  | 1.00      | 1.00   | 1.00 |
| Isabel     | 1.00      | 1.00   | 1.00 |
| Matías     | 1.00      | 1.00   | 1.00 |
| Vicky      | 1.00      | 1.00   | 1.00 |

Accuracy global: 1.00 (25/25).

Los scores de similitud individuales dieron entre 0.63 y 0.90 , nunca 1.0 exacto , lo que confirma que las imágenes de test no son duplicados ni imágenes demasiado similares a las del set de registro.

El score más bajo fue 0.6267 (Vicky), con un margen de 0.08 sobre el umbral configurado (0.55), el caso más ajustado de todo el test (casi falso negativo / unknown).

## 5. Robustez y generalización

Probé el sistema con transformaciones aplicadas sobre las 25 imágenes del test (flip horizontal, blur gaussiano, ruido gaussiano) para ver qué tan sólido es el matching:

| Transformación | Accuracy | Score promedio |
|-----------------|----------|-----------------|
| Original         | 1.00     | 0.7558          |
| Flip             | 1.00     | 0.7506          |
| Blur             | 1.00     | 0.7562          |
| Ruido            | 1.00     | 0.7120          |

El sistema identificó bien en las cuatro condiciones. El ruido gaussiano fue el que más bajó la similitud promedio, algo esperable porque afecta pixel a pixel y eso pega directo en las capas tempranas del backbone. El blur y el flip apenas movieron el embedding, lo que muestra que ArcFace aguanta bien este tipo de variaciones típicas de una captura real.

Como no hay entrenamiento de por medio, el data augmentation acá no se usó para entrenar nada, sino como forma de validar qué tan robusto es el matching frente a variaciones en la imagen de consulta.

## 6. Arquitectura del sistema

ArcFace recibe una imagen ya alineada y devuelve un embedding de 512 dimensiones. No sabe nada de identidades ni del conjunto de registro , es solo un extractor de características, no un clasificador.

La lógica de identificación toma ese embedding, lo compara contra todos los embeddings guardados en PostgreSQL (con pgvector) usando similitud coseno, se queda con el más parecido, y chequea si supera el umbral configurado (SIMILARITY_THRESHOLD=0.55 en el .env). Si lo supera, devuelve la etiqueta correspondiente; si no, devuelve "unknown".

Esto hace que una persona no registrada siempre dé "unknown", aunque esa persona exista en algún dataset público usado para pruebas , el sistema no clasifica sobre un conjunto fijo de clases, compara contra lo que esté efectivamente cargado en la base en ese momento.

Pipeline completo: detección (RetinaFace) → alineación (crop + landmarks, InsightFace) → embedding (ArcFace, 512D) → comparación (coseno contra pgvector) → identidad o "unknown".

## 7. Hiperparámetros

- det_size=(640, 640): tamaño de entrada al detector RetinaFace. Al principio tenía det_size=(112, 112) por error, heredado del tamaño de crop del reconocimiento. Con imágenes de detección tan chicas, el detector generaba tensores de escalas inconsistentes entre capas del feature pyramid, y tiraba ValueError: operands could not be broadcast together with shapes (18,) (32,) al decodificar los bounding boxes. Lo arreglé usando 640x640, que es el tamaño recomendado por InsightFace para buffalo_l.
- face_size=112: tamaño de entrada que espera el modelo de reconocimiento (w600k_r50), confirmado en el log de carga (recognition ['None', 3, 112, 112]). Coincide con el valor por defecto del .env de la cátedra.
- similarity_metric="cosine": similitud coseno, el estándar para embeddings ya normalizados como los de ArcFace.
- similarity_threshold=0.55: valor del .env de la cátedra. Lo validé con el test set: el score más bajo entre identificaciones correctas fue 0.6267, así que el margen es razonable pero no sobra mucho.

## 8. Problemas encontrados y soluciones

Race condition al levantar Postgres: el hook post_start del docker-compose.yml original de la cátedra corría CREATE EXTENSION vector apenas arrancaba el contenedor, sin esperar a que Postgres estuviera listo para aceptar conexiones. El timing no estaba garantizado y fallaba de forma intermitente con error de socket. Lo resolví cambiando el hook por depends_on: condition: service_healthy en el backend, sumado a un healthcheck con pg_isready en Postgres, y agregué un script de inicialización (docker-entrypoint-initdb.d) para crear la extensión sola en volúmenes nuevos.

Bug de det_size: ya lo conté en hiperparámetros. Lo encontré leyendo el traceback completo de InsightFace/RetinaFace, que señalaba el mismatch de shapes en distance2bbox.

LFW incompatible con el detector: también lo conté en la sección de dataset. Lo detecté mirando a mano una imagen que fallaba siempre (0 caras detectadas) y confirmando que el crop no tenía la cara completa.

Contaminación en el split train/test: la primera versión del split separaba solo en memoria, con listas de Python, y me daba un accuracy sospechosamente perfecto (scores de 1.0000 exactos). Comparando los embeddings de test contra la base encontré un caso con similitud 1.000000 entre una imagen de test y su propio crop, que ya estaba guardado en output/ de una corrida anterior con un dataset más chico. Lo resolví separando físicamente las carpetas de train y test en disco, y limpiando output/ de corridas previas antes de volver a registrar.

## 9. Modelo

No generé ni descargué ningún archivo de pesos propio (.pth), porque no hubo entrenamiento. Uso buffalo_l de InsightFace tal como se descarga del repositorio oficial (~280 MB) la primera vez que corre el pipeline, y queda cacheado en ~/.insightface/models/buffalo_l/ para las siguientes corridas.