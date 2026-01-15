import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(14, 10))
plt.subplot(3, 3, 1)
plt.imshow(binary, cmap='gray')
plt.title('Original Binary Image')
plt.axis('off')

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(binary, kernel, iterations=1)
plt.subplot(3, 3, 2)
plt.imshow(erosion, cmap='gray')
plt.title('Erosion')
plt.axis('off')

dilation = cv2.dilate(binary, kernel, iterations=1)
plt.subplot(3, 3, 3)
plt.imshow(dilation, cmap='gray')
plt.title('Dilation')
plt.axis('off')

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
plt.subplot(3, 3, 4)
plt.imshow(opening, cmap='gray')
plt.title('Opening (Remove Noise)')
plt.axis('off')

closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
plt.subplot(3, 3, 5)
plt.imshow(closing, cmap='gray')
plt.title('Closing (Fill Gaps)')
plt.axis('off')

gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
plt.subplot(3, 3, 6)
plt.imshow(gradient, cmap='gray')
plt.title('Morphological Gradient')
plt.axis('off')

kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

dilate_rect = cv2.dilate(binary, kernel_rect)
dilate_ellipse = cv2.dilate(binary, kernel_ellipse)
dilate_cross = cv2.dilate(binary, kernel_cross)

plt.subplot(3, 3, 7)
plt.imshow(dilate_rect, cmap='gray')
plt.title('Dilation - Rect Kernel')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(dilate_ellipse, cmap='gray')
plt.title('Dilation - Elliptical Kernel')
plt.axis('off')

plt.subplot(3, 3, 9)
plt.imshow(dilate_cross, cmap='gray')
plt.title('Dilation - Cross Kernel')
plt.axis('off')

plt.tight_layout()
plt.show()
