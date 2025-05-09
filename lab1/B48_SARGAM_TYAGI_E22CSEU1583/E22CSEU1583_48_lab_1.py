import cv2
import numpy as np
print(cv2.__version__)
img = cv2.imread("Picture1.jpg")
print(img.shape)
cv2.imshow('Image Window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
tx, ty = 50, 50  
matrix = np.float32([[1, 0, tx], [0, 1, ty]])
translated_image = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

cv2.imshow('Translated Image', translated_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
reflected_x = cv2.flip(img, 0)
reflected_y = cv2.flip(img, 1)

cv2.imshow('Reflected Image (X-axis)', reflected_x)
cv2.imshow('Reflected Image (Y-axis)', reflected_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
scaled_image = cv2.resize(img, None, fx=0.5, fy=1)

cv2.imshow('Scaled Image', scaled_image)
cv2.waitKey(10000)
cv2.destroyAllWindows()
x, y, width, height = 100, 100, 200, 200
cropped_image = img[y:y+height, x:x+width]

cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imwrite('resized_image.jpg', cropped_image)
resized_image = cv2.resize(img, (5, 1000))
cv2.imwrite("resize.jpg",resized_image)