import numpy as np 
import pandas as pd 
import keras
import cv2 as cv
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# get the dataset that was created on convert_to_csv.py
data_set = pd.read_csv(r'/home/greg/Documents/uni/AI/course_files/shuffledCK.csv', dtype='a')

# create two numpy arrays that will contain the data for emotion labels and image pixels separately 
emotion_labels = np.array(data_set['emotion'])
image_pixels = np.array(data_set['imgPixels'])

# convert numpy array to list

image_pixels = image_pixels.tolist()

# convert image_pixels array to 2D array, each array index can be displayed as an image using matplotlib
faces = []

for pixels in image_pixels:
    face = [int(pixel) for pixel in pixels.split(' ')]
    face = np.asarray(face).reshape(48, 48)
    face = cv.resize(face.astype('uint8'), (48, 48))
    faces.append(face.astype('float32'))

faces = np.asarray(faces)

# normalise the data to values between 0 and 1
faces /=255

# convert label array to one hot vector ie 4:'Sad' will become [0. 0. 0. 0. 1. 0. 0.]

emotion = keras.utils.to_categorical(emotion_labels, 7)

# split emotion data and appropriate labels to a train and test set equal to the number of images

train_size = 608


# prepare the image and label data to be used for the training phase

train_X = faces[0:train_size]
train_Y = emotion[0:train_size]
train_X = train_X.reshape(-1, 48, 48, 1)


# define the type of model as sequence of layers

model = Sequential()

##############################FEATURE_LEARNING##################################

# first layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second layer
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# third layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##############################FEATURE_LEARNING##################################


###############################CLASSIFICATION###################################

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

###############################CLASSIFICATION###################################

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history=model.fit(train_X, train_Y, validation_split=0.17, batch_size=10, epochs=20)


model.save('modelA.h5')

model.summary()

print(history.history.keys())

#plot model history data
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#plot model history loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



model.save('modelA.h5')

model.summary()



