import cv2
import numpy as np
import face_recognition
import os

# Load images from the folder
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Get name without extension

# Function to find encodings of images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList

# Encode known faces
encodeListKnown = findEncodings(images)
print("Encoding Complete")