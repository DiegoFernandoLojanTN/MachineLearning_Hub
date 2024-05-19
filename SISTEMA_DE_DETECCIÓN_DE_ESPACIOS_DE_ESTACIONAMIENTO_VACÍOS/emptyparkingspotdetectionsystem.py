import sys
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Configurar codificación UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Cargar el modelo guardado
model = load_model("emptyparkingspotdetectionmodel.h5")
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Nuevas coordenadas proporcionadas
coordinates = [
    [(204, 37), (350, 332)],
    [(351, 38), (492, 331)],
    [(493, 39), (635, 334)],
    [(636, 40), (778, 333)],
    [(780, 42), (922, 334)],
    [(920, 42), (1066, 335)],
    [(1066, 42), (1210, 333)],
    [(203, 369), (350, 663)],
    [(352, 369), (494, 663)],
    [(490, 370), (638, 664)],
    [(635, 372), (779, 665)],
    [(775, 370), (922, 669)],
    [(917, 376), (1068, 668)],
    [(1065, 370), (1212, 668)]
]

def detect_empty_parking(image, spot):
    x1, y1 = spot[0]
    x2, y2 = spot[1]
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Coordenadas no válidas para ROI")
        return False
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Vacio")
        return False
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (48,48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = np.expand_dims(resized_roi, axis=-1)
    prediction = model.predict(resized_roi)
    threshold = 0.01
    if prediction[0][0] > threshold:
        return True
    else:
        return False

# Cargar la imagen actual
current_image = cv2.imread("media/area.png")
if current_image is None:
    raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")

empty_count = 0

for spot in coordinates:
    if detect_empty_parking(current_image, spot):
        cv2.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)
        empty_count += 1
    else: 
        cv2.rectangle(current_image, spot[0], spot[1], (0,0,255), 2)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(current_image, f"Espacios vacíos: {empty_count}", (50,50), font, 1.5, (255,255,255), 3, cv2.LINE_AA)

cv2.imshow("Estacionamiento", current_image)
cv2.waitKey(0)
cv2.destroyAllWindows()