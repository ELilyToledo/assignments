import cv2
import numpy as np
import math


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read() 

    if not ret:
        print("Error, cannot recieve stream")
        break 
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
