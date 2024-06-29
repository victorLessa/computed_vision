import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
import os
from sklearn.preprocessing import OneHotEncoder

# Path for face image database
path = '../dataset'
recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

# function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples= list()
    names = list()
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L') # grayscale
        img_numpy = np.array(PIL_img,'uint8')
        name = os.path.split(imagePath)[-1].split(".")[2]
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            names.append(name.lower())
    # faceSamples = np.asarray(faceSamples, np.uint8)
    return faceSamples,names

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces,names = getImagesAndLabels(path)

# Treinando

uniq_names = list(set(names))

labels = []
for indice, name in enumerate(names):
    for unique_indice, uniq_name  in enumerate(uniq_names):
        if uniq_name == name:
            labels.append(unique_indice)

recognizer.train(faces, np.array(labels))

# # Save the model into trainer/trainer.yml
recognizer.write('./trainer.yml') 
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(names))))