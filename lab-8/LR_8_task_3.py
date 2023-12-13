import cv2
import numpy as np
img = cv2.imread("me.jpg")
print(img.shape)
imgResize = cv2.resize(img, (1000, 500))
print(imgResize.shape)
imgCropped = img[150:800, 60:700]
cv2.imshow("Image", img)
# cv2.imshow("Image Resize",imgResize)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)