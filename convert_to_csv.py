import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

''' This script will allow us to prepare the images within the Cohne Kanade dataset by detecting faces and transfering the pixel data images and image labels to a csv file'''

# initialise 2D arrays for the image data to parse into and an array for the the label data
image_data = np.zeros([608, 48*48], dtype=np.uint8)
image_labels = np.zeros([608], dtype=int)


# required by opencv to detect faces on the image
cascade = cv2.CascadeClassifier("/home/greg/Documents/uni/AI/AI_emotion/haarcascade_frontalface_alt.xml")

# location of the Cohne Kanade images
DATASET = "/home/greg/Documents/uni/AI/CKImages"

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# using os library to traverse through the dataset
filesystem = os.listdir(DATASET)


# function that uses opencv to return the roi (face) without the unecessary image background
def detect_face(image, cascade):
    face = cascade.detectMultiScale(image, scaleFactor = 1.1, minNeighbors =10, minSize=(5,5), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x,y,w,h) in face:
        roi = image[y:y+h, x:x+w]
        return roi


#main function to process the dataset
def process_dataset():

    count = 0

    for emotion in EMOTIONS:

        path = os.path.join(DATASET, emotion)
        for file_name in os.listdir(path):
            if not file_name.startswith('.') and os.path.isfile(os.path.join(path, file_name)):
                image = cv2.imread(os.path.join(path, file_name))
                print("processing " + path, file_name)

                #convert image to grayscale to reduce complexity
                image2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                face = detect_face(image2gray, cascade)
                face_reduce = cv2.resize(face, (48, 48))

                if(emotion == 'anger'):
                    image_labels[count] = 0

                elif(emotion == 'disgust'):
                    image_labels[count] = 1

                elif(emotion == 'fear'):
                    image_labels[count] = 2

                elif(emotion == 'joy'):
                    image_labels[count] = 3

                elif(emotion == 'neutral'):
                    image_labels[count] = 4

                elif(emotion == 'sadness'):
                    image_labels[count] = 5

                elif(emotion == 'surprise'):
                    image_labels[count] = 6

                else:
                    print("error at label switch statement")
                #parse pixel to the 2D array and then flatten converts the array to 1D
                image_data[count][0:48*48] = np.ndarray.flatten(face_reduce)
                count+=1



process_dataset()


#write numpy arrays to csv file
with open (r"CK_DATA.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['emotion','imgPixels'])
    for i in range(len(image_labels)):
        data_list = list(image_data[i])
        label = " ".join(str(x) for x in data_list)
        row = np.hstack([image_labels[i], l])
        writer.writerow(b)
