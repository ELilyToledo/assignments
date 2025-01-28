import cv2
import numpy as np

def circledetection(frame):
    h,w,_ = frame.shape
    aratio = w/h

    #resizing the image by the aspect ratio
    resize = cv2.resize(frame, (500, int(500 / aratio)))
    copy = resize.copy()

    #mask = np.zeros(img.shape[:2], dtype="uint8")

    #applying grayscale and blurring the image
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    #cv2.imshow("blur",blur)

    #finding circles using hough
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 900, param1=60, param2=110)

    if circles is not None:
        #rounding the values in the numpy array
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            x,y,r = circle
            #draw circle and center point
            cv2.circle(copy, (x,y), r, (0,255,0), 3)
            cv2.circle(copy, (x, y), 1, (0, 255, 0), 2)
            #cv2.circle(mask, (x, y), r, 255, -1)

    #masked = cv2.bitwise_and(img, img, mask=mask)
    #cv2.imshow("masked", mask)

    return copy

cap = cv2.VideoCapture("tube3crop.mp4")

#loading the video file
if (cap.isOpened() == False):
    print("Cannot open video")

while (cap.isOpened()):

    ret, frame = cap.read()
    if ret == True:

        #apply overlay
        copy = circledetection(frame)
        cv2.imshow('frame', copy)

        #click q to exit video
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
