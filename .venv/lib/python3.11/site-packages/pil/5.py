import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

edges = cv2.Canny(img, 100, 200)

plt.subplot(2, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

ret1, th_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.subplot(2, 3, 3)
plt.imshow(th_global, cmap='gray')
plt.title(f'Global Thresholding (T={ret1})')
plt.axis('off')

th_adapt_mean = cv2.adaptiveThreshold(img, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

plt.subplot(2, 3, 4)
plt.imshow(th_adapt_mean, cmap='gray')
plt.title('Adaptive Thresholding (Mean)')
plt.axis('off')

th_adapt_gauss = cv2.adaptiveThreshold(img, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

plt.subplot(2, 3, 5)
plt.imshow(th_adapt_gauss, cmap='gray')
plt.title('Adaptive Thresholding (Gaussian)')
plt.axis('off')

plt.tight_layout()
plt.show()