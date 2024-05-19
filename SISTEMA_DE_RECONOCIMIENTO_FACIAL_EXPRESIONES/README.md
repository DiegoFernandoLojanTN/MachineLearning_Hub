# Detector de Emociones con Redes Neuronales

Este proyecto implementa un sistema de detección de emociones a partir de imágenes de rostros utilizando redes neuronales convolucionales (CNN) y técnicas de visión por computadora.

## Descripción del Proyecto

El objetivo del proyecto es clasificar imágenes de rostros en una de las siete emociones básicas: enojado, asco, miedo, feliz, neutral, triste y sorpresa. Se utiliza un modelo de CNN entrenado en el [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset).

## Tecnologías Utilizadas

- **Redes Neuronales Convolucionales (CNN)**: Para el procesamiento y clasificación de imágenes.
- **Visión por Computadora**: Utilizando OpenCV para la detección de rostros en tiempo real.
- **Keras y TensorFlow**: Para la construcción y entrenamiento del modelo de CNN.
- **OpenCV**: Para la captura y preprocesamiento de imágenes.
- **Matplotlib**: Para la visualización de resultados.

## Estructura del Proyecto

### Notebook

El archivo `Detector_Emociones_RNC.ipynb` incluye:

1. **Carga y Preprocesamiento del Dataset**: Preparación de datos de entrenamiento y validación.
2. **Construcción del Modelo**: Definición y compilación de la arquitectura de la CNN.
3. **Entrenamiento y Evaluación**: Entrenamiento del modelo y evaluación de su desempeño.

### Script

El archivo `Emotions.py` incluye:

1. **Detección de Rostros en Tiempo Real**: Utilización de un modelo preentrenado de OpenCV para detectar rostros.
2. **Clasificación de Emociones**: Uso del modelo entrenado para clasificar la emoción del rostro detectado.
3. **Visualización de Resultados**: Visualización de las emociones detectadas en tiempo real.

## Resultados

### Ejemplos de Detección en Tiempo Real

A continuación se muestran algunos ejemplos de detección de emociones en tiempo real utilizando la webcam:

[![Detección de Emociones 1](https://i.postimg.cc/T1pH1k9f/Captura-de-pantalla-65.png)](https://postimg.cc/gLbKBHQT)
*Descripción: Detección de emoción*


## Contribuciones

Las contribuciones a este proyecto son bienvenidas. Puedes hacerlo a través de pull requests o abriendo issues para reportar problemas y sugerencias.


