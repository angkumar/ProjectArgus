import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img-original4.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Original Image', img)
cv2.waitKey(0)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 4))
plt.title('Original Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.show()

equalized = cv2.equalizeHist(img)
cv2.imshow('Equalized Image', equalized)
cv2.waitKey(0)

hist_eq = cv2.calcHist([equalized], [0], None, [256], [0, 256])

plt.figure(figsize=(10, 4))
plt.title('Equalized Image Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.plot(hist_eq, color='black')
plt.xlim([0, 256])
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.plot(hist, color='black')
plt.title('Original Histogram')
plt.xlim([0, 256])

plt.subplot(2, 2, 3)
plt.imshow(equalized, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.plot(hist_eq, color='black')
plt.title('Equalized Histogram')
plt.xlim([0, 256])

plt.tight_layout()
plt.show()

cv2.destroyAllWindows()