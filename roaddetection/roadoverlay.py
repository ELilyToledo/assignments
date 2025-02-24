import cv2
import numpy as np


def displayarrow(frame):
    # upload and set each transparent png to an arrow
    upar = cv2.imread('arrows/uparrow.png', cv2.IMREAD_UNCHANGED)
    downar = cv2.imread('arrows/downarrow.png', cv2.IMREAD_UNCHANGED)
    rightar = cv2.imread('arrows/rightarrow.png', cv2.IMREAD_UNCHANGED)
    leftar = cv2.imread('arrows/leftarrow.png', cv2.IMREAD_UNCHANGED)

    #resize the png
    arrow = upar
    h, w, _ = arrow.shape
    arrow = cv2.resize(arrow, (int(h*0.3), int(w*0.3)))
    
    arrbgr = arrow[:, :, :3]
    arralph = arrow[:, :, 3]

    #set the coordinates of where the arrow will be
    roix, roiy = 1725, 35
    h, w = arrbgr.shape[:2]

    if roiy + h <= frame.shape[0] and roix + w <= frame.shape[1]:
        roi = frame[roiy:roiy + h, roix:roix + w]
        mask = arralph.astype(float) / 255.0

    for x in range(3):
        roi[:, :, x] = (mask * arrbgr[:, :, x] + (1 - mask) * roi[:, :, x])

    return frame


def overlay(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 90)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (np.ones((21, 21), np.uint8)))

    cv2.imshow("edge", edges)
    cv2.imshow("closed", closed)


cap = cv2.VideoCapture("roadvid.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = displayarrow(frame)
    overlay(frame)

    cv2.imshow("Curved Path Overlay", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
