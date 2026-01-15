import cv2
from PIL import Image

img_cv = cv2.imread("img-original4.jpg")
cv2.imshow("Original Image (OpenCV)", img_cv)
cv2.waitKey(0)

img_pil = Image.open("img-original4.jpg")
img_pil.show()

resized_cv = cv2.resize(img_cv, (300, 300))
cv2.imshow("Resized Image (OpenCV)", resized_cv)
cv2.waitKey(0)

resized_pil = img_pil.resize((300, 300))
resized_pil.show()

cropped_cv = img_cv[100:400, 100:400]
cv2.imshow("Cropped Image (OpenCV)", cropped_cv)
cv2.waitKey(0)

cropped_pil = img_pil.crop((100, 100, 400, 400))
cropped_pil.show()

(h, w) = img_cv.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_cv = cv2.warpAffine(img_cv, matrix, (w, h))
cv2.imshow("Rotated Image (OpenCV)", rotated_cv)
cv2.waitKey(0)

rotated_pil = img_pil.rotate(45)
rotated_pil.show()

zoom_factor = 1.5
zoomed_cv = cv2.resize(img_cv, None, fx=zoom_factor, fy=zoom_factor)
cv2.imshow("Zoomed Image (OpenCV)", zoomed_cv)
cv2.waitKey(0)

width, height = img_pil.size
zoomed_pil = img_pil.resize((int(width * 1.5), int(height * 1.5)))
zoomed_pil.show()

shrink_factor = 0.5
shrinked_cv = cv2.resize(img_cv, None, fx=shrink_factor, fy=shrink_factor)
cv2.imshow("Shrunk Image (OpenCV)", shrinked_cv)
cv2.waitKey(0)

shrunk_pil = img_pil.resize((int(width * 0.5), int(height * 0.5)))
shrunk_pil.show()

flip_horizontal_cv = cv2.flip(img_cv, 1)
flip_vertical_cv = cv2.flip(img_cv, 0)
cv2.imshow("Flipped Horizontally (OpenCV)", flip_horizontal_cv)
cv2.imshow("Flipped Vertically (OpenCV)", flip_vertical_cv)
cv2.waitKey(0)

flip_horizontal_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
flip_vertical_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
flip_horizontal_pil.show()
flip_vertical_pil.show()

cv2.destroyAllWindows()