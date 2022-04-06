from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import Normalizer
import cv2
import keyboard

"""person = ['daniel', 'unknown', 'jacirene', 'vagner', 'valmir', 'vigner'] As classes devem ser declaradas de acordo 
com a ordem realizada no treinamento"""
person = [] #declarar as classes 
capture = cv2.VideoCapture(0)
# num_classes = len(person)

face_detection = MTCNN()
facenet_keras = load_model('facenet.h5') # rede convolucional Facenet 
modelo = load_model('face_6classes_final.h5') # especificar modelo gerado a partir do treinamento no formato .h5

def carreg_embedding(facenet_keras, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    expand = np.expand_dims(face_pixels, axis=0)
    yhat = facenet_keras.predict(expand)

    return yhat[0]


def extracao_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)

    return np.asarray(image)

while capture.isOpened():

    config, frame = capture.read()
    faces = face_detection.detect_faces(frame)

    for face in faces:

        confianca = face['confidence'] * 100

        if confianca > 97.5:
            x1, y1, w, h = face['box']
            face = extracao_face(frame, face['box'])

            # processo de normalização
            face = face.astype('float32') / 255
            emb = carreg_embedding(facenet_keras, face)
            tens = np.expand_dims(emb, axis=0)
            norm = Normalizer(norm='l2')

            tensor = norm.transform(tens)
            classe = np.argmax(modelo.predict(tensor), axis=-1)[0]
            probabilities = modelo.predict(tensor)
            probabilities = probabilities[0][classe] * 100

            if probabilities > 97.5:
                user = str(person[classe]).upper()
                color = (76, 153, 0) # Cor na sequência/formato BGR
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.6
                cv2.putText(frame, user, (x1, y1 - 10), font, fontScale=font_size, color=color, thickness=2)

    cv2.imshow('Capturing', frame)
    key = cv2.waitKey(1)

    if keyboard.is_pressed('q'):
        # if key == 9:
        break

capture.release()
cv2.destroyAllWindows()
