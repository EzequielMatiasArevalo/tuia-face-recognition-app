# Documentación del Dataset

Este documento describe la composición y los criterios utilizados para la construcción del dataset de reconocimiento facial.

## Reglas de Composición

1. **Cantidad por persona:** Se han tomado **10 imágenes** por cada individuo presente en el dataset.
2. **Excepción por desbalanceo (Ambrogi):** A diferencia del resto, para la clase correspondiente a *Ambrogi* se han incluido **todas las imágenes disponibles**. El objetivo de esta excepción es probar el comportamiento y la robustez del modelo frente a un desbalance de clases significativo.
3. **Excepción de exclusión (Valentino):** Para el caso de *Valentino*, se tomó la decisión deliberada de **no incluir ninguna imagen (0 imágenes)** en el dataset de entrenamiento/registro. 
   - *Propósito:* Dado que Valentino es el hermano de Gianluca (quien sí forma parte del dataset), esto permite probar el sistema con una persona "externa" (*unknown*) y evaluar si la red neuronal asigna una mayor similitud con Gianluca debido a los rasgos faciales compartidos, en comparación con el resto de las personas del dataset.
