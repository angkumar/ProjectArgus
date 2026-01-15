import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

average_blur = cv2.blur(img, (5, 5))

plt.subplot(2, 3, 2)
plt.imshow(average_blur)
plt.title('Averaging Filter')
plt.axis('off')

gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)

plt.subplot(2, 3, 3)
plt.imshow(gaussian_blur)
plt.title('Gaussian Filter')
plt.axis('off')

sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
sharpened = cv2.filter2D(img, -1, sharpen_kernel)

plt.subplot(2, 3, 4)
plt.imshow(sharpened)
plt.title('Sharpened Image')
plt.axis('off')

sharpen_kernel2 = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
sharpened_strong = cv2.filter2D(img, -1, sharpen_kernel2)

plt.subplot(2, 3, 5)
plt.imshow(sharpened_strong)
plt.title('Strong Sharpening')
plt.axis('off')

plt.tight_layout()
plt.show()
