import cv2
import numpy as np

img = cv2.imread('sodacantopimage.png')

h,w,_ = img.shape
aratio = w/h

#resizing the image by the aspect ratio
resize = cv2.resize(img, (600, int(600 / aratio)))
img = resize.copy()

#mask = np.zeros(img.shape[:2], dtype="uint8")

#applying grayscale and blurring the image
gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 0)
cv2.imshow("blur",blur)

#finding circles using hough
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 170, param1=50, param2=100)

if circles is not None:
    #rounding the values in the numpy array
    circles = np.round(circles[0, :]).astype("int")
    for circle in circles:
        x,y,r = circle
        cv2.circle(img, (x,y), r, (0,255,0), 2)
        cv2.circle(img, (x, y), 1, (0, 255, 0), 2)
        #cv2.circle(mask, (x, y), r, 255, -1)

#masked = cv2.bitwise_and(img, img, mask=mask)
#cv2.imshow("masked", mask)

cv2.imshow('circles', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
