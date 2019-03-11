#!/usr/bin/python
# coding:utf8
import numpy as np
import pandas as pd
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#emotion ={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

data = pd.read_csv(r'/home/greg/Documents/uni/AI/course_files/shuffledCK.csv', dtype='a')
label = np.array(data['emotion'])
img_data = np.array(data['imgPixels'])
N_sample = label.size
Face_data = np.zeros((N_sample, 48*48))
Face_label = np.zeros((N_sample, 7), dtype=np.float)


for i in range(N_sample):
    x = img_data[i]
    x = np.fromstring(x, dtype=float, sep=' ')
    #normalise the data to float between 0 and 1
    x = x/255
    Face_data[i] = x
    Face_label[i, int(label[i])] = 1.0

train_num = 500
test_num = 108

train_x = Face_data [0:train_num, :]
train_y = Face_label [0:train_num, :]
train_x = train_x.reshape(-1,48,48,1)

test_x = Face_data [train_num : train_num+test_num, :]
test_y = Face_label [train_num : train_num+test_num, :]
test_x = test_x.reshape(-1,48,48,1)

model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


datagen = ImageDataGenerator(
                            featurewise_center=True,
                            featurewise_std_normalization=True,
                            rotation_range=20,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            horizontal_flip=True)
datagen.fit(train_x)
model.fit_generator(datagen.flow(train_x, train_y, batch_size=10),steps_per_epoch=len(train_x), epochs=20)

model.fit(train_x, train_y, batch_size=10, epochs=80)
score = model.evaluate(test_x, test_y, batch_size=10)

model.save('my_newm.h5')

model.summary()
