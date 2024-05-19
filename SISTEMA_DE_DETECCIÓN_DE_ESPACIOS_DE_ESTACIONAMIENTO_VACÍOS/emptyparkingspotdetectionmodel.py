import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Actualiza las rutas a las carpetas correctas
training_data = [
    "C:/Users/ferna/CursoMachine/car_detection/media/matchbox_cars_parkinglot/empty",     # Cambia esta ruta a la ruta correcta en tu sistema
    "C:/Users/ferna/CursoMachine/car_detection/media/matchbox_cars_parkinglot/occupied"   # Cambia esta ruta a la ruta correcta en tu sistema
]

def load_images(training_data):
    images = []
    labels = []
    for i, folder in enumerate(training_data):
        label = i
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist.")
            continue
        if not os.listdir(folder):
            print(f"Warning: Folder {folder} is empty.")
            continue
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(label)
                else:
                    print(f"Warning: {img_path} is not a file.")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Cargar imágenes y etiquetas
images, labels = load_images(training_data)

# Verificar si se cargaron imágenes
if len(images) == 0:
    raise ValueError("No images found. Please check the folder paths and ensure they contain images.")

# División de los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar las imágenes
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Aumentación de datos
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# Definir el modelo
model = Sequential()
model.add(Input(shape=(48, 48, 1)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Compilar el modelo
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Entrenar el modelo
model.fit(datagen.flow(X_train, y_train, batch_size=64),
          steps_per_epoch=len(X_train) // 64, epochs=20,
          validation_data=(X_test, y_test))

# Guardar el modelo
model.save("emptyparkingspotdetectionmodel.h5")