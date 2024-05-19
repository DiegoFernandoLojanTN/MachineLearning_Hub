# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x = [0,1,2,3,4,5,6] 
y= [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
plt.ion() 
figura1 = plt.figure()
bar1 = plt.bar(x,y)
#my_colors = 'rgbykmc'
my_colors = ['red', 'green', 'blue', 'orange', 'purple', 'yellow', 'cyan']

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones del detector
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']

# Cargamos el  modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("detectionfacesemotion.h5")

# Se crea la captura de video
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# Toma la imagen, los modelos de detección de rostros y mascarillas 
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar rostros, ubicaciones y predicciones
    faces = []
    locs = []
    preds = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Comprobar si la región de interés está dentro de los límites de la imagen original
            if Yi >= 0 and Yf <= frame.shape[0] and Xi >= 0 and Xf <= frame.shape[1]:
                face = frame[Yi:Yf, Xi:Xf]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face2 = img_to_array(face)
                face2 = np.expand_dims(face2, axis=0)

                # Se agregan los rostros y las localizaciones a las listas
                faces.append(face2)
                locs.append((Xi, Yi, Xf, Yf))
                preds.append(emotionModel.predict(face2)[0])

    return locs, preds


while True:
	# Se toma un frame de la cámara y se redimensiona
	ret, frame = cam.read()
	frame = imutils.resize(frame, width=640)

	(locs, preds) = predict_emotion(frame,faceNet,emotionModel)
	
	# Para cada hallazgo se dibuja en la imagen el bounding box y la clase
	for (box, pred) in zip(locs, preds):
		
		(Xi, Yi, Xf, Yf) = box
		(angry,disgust,fear,happy,neutral,sad,surprise) = pred


		label = ''
		# Se agrega la probabilidad en el label de la imagen
		label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry,disgust,fear,happy,neutral,sad,surprise) * 100)

		cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255,0,0), -1)
		cv2.putText(frame, label, (Xi+5, Yi-15),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
		cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255,0,0), 3)

		plt.clf()
		y = [angry,disgust,fear,happy,neutral,sad,surprise]
		plt.xticks(x, ['enojado', 'asco', 'miedo', 'feliz', 'neutral', 'triste', 'sorpresa'])
		plt.grid(True)
		plt.ylim([0.0,1.0])
		#plt.bar(x,y,color=my_colors,width=1)
		plt.bar(x, y, color=my_colors, width=1)

		figura1.canvas.draw()

	time_actualframe = time.time()

	if time_actualframe>time_prevframe:
		fps = 1/(time_actualframe-time_prevframe)
	
	time_prevframe = time_actualframe

	cv2.putText(frame, str(int(fps))+" FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

	cv2.imshow("Frame", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		 break

cv2.destroyAllWindows()
cam.release()