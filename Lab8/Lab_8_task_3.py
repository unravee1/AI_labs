import cv2

img = cv2.imread("me.jpg")
print(img.shape)
imgResize = cv2.resize(img, (350, 470))
print(imgResize.shape)
imgCropped = img[10:450, 50:350]
cv2.imshow("Image", img)
cv2.imshow("Image Cropped", imgCropped)
cv2.waitKey(0)