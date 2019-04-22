from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image

root = Tk()

from keras.models import load_model
model = load_model('modelA.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')


path = filedialog.askopenfilename(initialdir = "/home/greg/Desktop/reporTest", title = "Select file", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))

print(path)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detect_face(image, face_cascade):
    face = face_cascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors = 10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)

    for(x,y,w,h) in face:
        roi = image[y:y+h, x:x+w]
        return roi






img = cv2.imread(path)

image2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face = detect_face(image2gray, face_cascade)

face_reduce = cv2.resize(face, (48, 48))

img_pix = image.img_to_array(face_reduce)
img_pix = np.expand_dims(img_pix, axis = 0)

img_pix /=255

predictions = model.predict(img_pix)



max_index = np.argmax(predictions[0])

top_pred = np.amax(predictions)

rec = (emotions[max_index])

print("Model Predicted " + rec + " " + "{:.2%}".format(top_pred))

print("##############SUMMARY###############")

newList =[]

for i in range(7):

    if i < 1:
        newList.append('%f' % predictions[0][:i+1])
    elif i >= 1 and i < 6:
        newList.append('%f' % predictions[0][i:i+1])
    elif i == 6:
        newList.append('%f' % predictions[0][i:])


newList = [float(i) for i in newList]

for i in range(7):

    print(emotions[i] + ": " + "{:.2%}".format(newList[i]))

print("####################################")

