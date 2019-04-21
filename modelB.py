import numpy as np 
import pandas as pd 
import keras
import cv2 as cv
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

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

train_size = 500
test_size = 108

# prepare the image and label data to be used for the training phase

train_X = faces[0:train_size]
train_Y = emotion[0:train_size]
train_X = train_X.reshape(-1, 48, 48, 1)

# prepare the image and label data to be used for the testing phase from index 500 to the rest

test_X = faces[train_size:608]
test_Y = emotion[train_size:608]
test_X = test_X.reshape(-1, 48, 48, 1)

# define the type of model as sequence of layers

model = Sequential()

##############################FEATURE_LEARNING##################################

# first layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))

# second layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))


##############################FEATURE_LEARNING##################################


###############################CLASSIFICATION###################################

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax'))

###############################CLASSIFICATION###################################

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

datagen = ImageDataGenerator(
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)
datagen.fit(train_X)

model.fit_generator(datagen.flow(train_X, train_Y, batch_size=10),steps_per_epoch=len(train_X), epochs=20)


model.fit(train_X, train_Y, batch_size=10, epochs=80)

train_score = model.evaluate(train_X, train_Y, batch_size=10)
print('Train Loss', train_score[0])
print('Train accuracy', 100*train_score[1])

test_score = model.evaluate(test_X, test_Y, batch_size=10)
print('Test Score', test_score[0])
print('Test accuracy', 100*test_score[1])


model.save('finalModel2.h5')

model.summary()



