from tkinter import filedialog
from tkinter import *
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image

root = Tk()

from keras.models import load_model
model = load_model('my_newm.h5')

emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')


path = filedialog.askopenfilename(initialdir = "/home/greg/Downloads/GoogleSet", title = "Select file", filetypes = (("jpeg files", "*.jpg"), ("all files", "*.*")))

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

print(predictions)

top_pred = np.amax(predictions)

rec = (emotions[max_index])

print("predicted " + rec + " " + "{:.2%}".format(top_pred))

print("############################")

newList =[]

newList.append(predictions[0][:1])
newList.append(predictions[0][1:2])
newList.append(predictions[0][2:3])
newList.append(predictions[0][3:4])
newList.append(predictions[0][4:5])
newList.append(predictions[0][5:6])
newList.append(predictions[0][6:])


for l in newList:
    print(l)

#predictions = model.predict(text[0])
#max_index = np.argmax(predictions[0])
#emotion = emotions[max_index]

#print(np.amax(predictions)*100)

