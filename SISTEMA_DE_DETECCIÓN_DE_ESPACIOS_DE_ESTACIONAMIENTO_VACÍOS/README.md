# Sistema de Detección de Espacios de Estacionamiento Vacíos

Este proyecto implementa un sistema de visión por computadora para detectar espacios de estacionamiento vacíos en una imagen de un estacionamiento. Utiliza técnicas de aprendizaje profundo y procesamiento de imágenes para identificar si un espacio de estacionamiento está ocupado o vacío.

## Descripción Técnica

### Primer Código: Entrenamiento del Modelo

El primer script se encarga de cargar las imágenes de entrenamiento, procesarlas y entrenar un modelo de red neuronal convolucional (CNN) para clasificar las imágenes en espacios de estacionamiento vacíos u ocupados.

#### Pasos:

1. **Importación de Bibliotecas**: Se importan las bibliotecas necesarias para el procesamiento de imágenes y el entrenamiento del modelo.

2. **Carga de Imágenes**: Se define una función para cargar las imágenes desde las rutas especificadas, redimensionarlas y asignarles etiquetas. Las imágenes se cargan desde carpetas específicas que contienen ejemplos de espacios vacíos y ocupados.

3. **Procesamiento y Normalización**: Las imágenes y etiquetas se dividen en conjuntos de entrenamiento y prueba. Las imágenes se normalizan para mejorar el rendimiento del modelo.

4. **Aumentación de Datos**: Se aplican técnicas de aumentación de datos para incrementar la diversidad del conjunto de entrenamiento, lo cual ayuda a evitar el sobreajuste del modelo.

5. **Definición y Entrenamiento del Modelo**: Se define la arquitectura de la red neuronal utilizando capas convolucionales, de pooling, y de dropout para mejorar la precisión y evitar el sobreajuste. El modelo se entrena con los datos procesados.

6. **Guardar el Modelo**: El modelo entrenado se guarda para su uso posterior en la detección de espacios vacíos.

#### Imágenes Sugeridas

- **Imágenes de Ejemplo de Entrenamiento**: Incluye algunas imágenes de ejemplo de los datos de entrenamiento, tanto de espacios vacíos como ocupados.

    [![Espacio Ocupado](https://i.postimg.cc/0QVZzMVv/roi-51bf456245e349a29880097e0834a040-occupied.jpg)](https://postimg.cc/tZVFfT0v)
    *Ejemplo de imagen de espacio ocupado*
    
    [![Espacio Vacío](https://i.postimg.cc/NGJDkXB2/roi-e7292c8ea3654a88baa82e39871911af-empty.jpg)](https://postimg.cc/0rS78b69)
    *Ejemplo de imagen de espacio vacío.*

### Segundo Código: Detección en Tiempo Real

El segundo script carga el modelo entrenado y lo utiliza para detectar espacios de estacionamiento vacíos en una imagen nueva.

#### Pasos:

1. **Importación de Bibliotecas y Configuración**: Se importan las bibliotecas necesarias y se configura la codificación UTF-8 para asegurar la correcta visualización de caracteres.

2. **Cargar el Modelo y la Imagen**: Se carga el modelo previamente guardado y la imagen actual del área de estacionamiento. Esta imagen se utiliza para detectar los espacios vacíos.

3. **Definición de Coordenadas y Función de Detección**: Se definen las coordenadas de los espacios de estacionamiento en la imagen y se implementa una función para detectar si un espacio está vacío.

4. **Detección y Visualización**: Se detectan los espacios vacíos y se dibujan rectángulos de diferentes colores dependiendo de si el espacio está vacío u ocupado. La imagen resultante se muestra al usuario.

#### Imágenes Sugeridas

- **Imagen Original del Estacionamiento**: Incluye una imagen del área de estacionamiento antes de la detección.

    [![Estacionamiento Original](https://i.postimg.cc/CL4vXKyb/area.png)](https://postimg.cc/F77gJNHK)
    *Imagen del área de estacionamiento antes de la detección.*

- **Imagen con Detecciones**: Incluye una imagen del área de estacionamiento después de la detección, mostrando los espacios vacíos y ocupados.

    [![Detección de Espacios](https://i.postimg.cc/d3V70n1s/imagen-2024-05-19-155407702.png)](https://postimg.cc/JDS76NZv)
    *Imagen del área de estacionamiento después de la detección, con los espacios vacíos en verde y los ocupados en rojo.*

## Cómo Ejecutar

1. **Revisa la ruta de los recursos**: En la pagina principal se encuentra en enlace con todos los recursos.
1. **Entrenar el Modelo**: Ejecuta el primer script para entrenar el modelo.
2. **Detectar Espacios Vacíos**: Ejecuta el segundo script para utilizar el modelo entrenado y detectar espacios vacíos en una nueva imagen.

