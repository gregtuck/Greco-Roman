import numpy as np
import cv2
from keras.preprocessing import image

face_cascade = cv2.CascadeClassifier("/home/greg/Documents/uni/AI/AI_emotion/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
#-----------------------------
#face expression recognizer initialization

from  keras.models import load_model
model = load_model('my_newm.h5')

#-----------------------------

emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

while(True):

    ret, img = cap.read()
	#img = cv2.imread('C:/Users/IS96273/Desktop/hababam.jpg')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #print(faces) #locations of detected faces

    for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(209,66,244),2) #draw rectangle to main image

	    detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
	    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
	    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

	    img_pixels = image.img_to_array(detected_face)
	    img_pixels = np.expand_dims(img_pixels, axis = 0)

	    img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]



	    predictions = model.predict(img_pixels) #store probabilities of 7 expression
	    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
	    max_index = np.argmax(predictions[0])
	    emotion = emotions[max_index]
	   # write emotion text above rectangle
	    cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_DUPLEX, 1, (57,244,0), 2)
	    #process on detected face end
	    #-----------------------

    print(max_index)
    value = np.amax(predictions)
    print(value)

    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
        break

#kill open cv things
cap.release()
cv2.destroyAllWindows()
