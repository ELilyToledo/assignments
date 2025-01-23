import cv2
import numpy as np

def circledetection(frame):
    h,w,_ = frame.shape
    aratio = w/h

    #resizing the image by the aspect ratio
    resize = cv2.resize(frame, (600, int(600 / aratio)))
    copy = resize.copy()

    #mask = np.zeros(img.shape[:2], dtype="uint8")

    #applying grayscale and blurring the image
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (17, 17), 0)
    blur2 = cv2.medianBlur(gray, 13)
    #cv2.imshow("blur",blur)

    #finding circles using hough
    circles = cv2.HoughCircles(blur2, cv2.HOUGH_GRADIENT, 1, 170, param1=50, param2=100)

    if circles is not None:
        #rounding the values in the numpy array
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            x,y,r = circle
            cv2.circle(copy, (x,y), r, (0,255,0), 2)
            cv2.circle(copy, (x, y), 1, (0, 255, 0), 2)
            #cv2.circle(mask, (x, y), r, 255, -1)

    #masked = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("masked", mask)

    return copy

cap = cv2.VideoCapture("circlevid.mov")

if (cap.isOpened() == False):
    print("Error opening video file")

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:

        copy = circledetection(frame)
        cv2.imshow('Frame', copy)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
