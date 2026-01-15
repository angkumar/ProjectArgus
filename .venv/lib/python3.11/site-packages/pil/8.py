import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def convertToRGB(img):
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

image = cv.imread('sample.jpg')
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.11, minNeighbors=5)

for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(convertToRGB(image))
plt.title(f"Faces detected: {len(faces)}")
plt.axis('off')
plt.show()

print("Number of faces detected:", len(faces))