import cv2
# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("me.jpg")
# DISPLAY
cv2.imshow("Result", img)
cv2.waitKey(0)
