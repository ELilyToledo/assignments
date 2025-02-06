import cv2
import numpy as np


def curvedetection(frame, roi):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (np.ones((43, 43), np.uint8)))

    cv2.imshow("edge", edges)

    rx, ry, rw, rh = roi
    mask = np.zeros_like(closed)
    mask[ry:ry + rh, rx:rx + rw] = closed[ry:ry + rh, rx:rx + rw]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filconts = []

    for contour in contours:
        if cv2.arcLength(contour, True) > 100:
            filconts.append(contour)

    if len(filconts) >= 2:
        filconts = sorted(filconts, key=cv2.contourArea, reverse=True)
        cont1 = filconts[0]
        cont2 = filconts[1]

        poly1 = cv2.approxPolyDP(cont1, 8, closed=True)
        poly2 = cv2.approxPolyDP(cont2, 8, closed=True)

        poly1 = poly1[:int(len(poly1) * 0.5)]
        poly2 = poly2[:int(len(poly2) * 0.6)]

        cv2.polylines(frame, [poly1], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [poly2], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        centerline = []
        for p1, p2 in zip(poly1, poly2):
            midpoint = ((p1[0][0] + p2[0][0]) // 2, (p1[0][1] + p2[0][1]) // 2)
            centerline.append(midpoint)

        # centerline.append(poly1[-1])
        if len(centerline) > 0:
            centerline = np.array(centerline)
            centerline = cv2.approxPolyDP(centerline, epsilon=8, closed=True)
            cv2.polylines(frame, [centerline], isClosed=False, color=(255, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    return frame


cap = cv2.VideoCapture(0)

roi = (200, 100, 900, 800)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    overlay = curvedetection(frame, roi)
    cv2.imshow("Curved Path Detection", overlay)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
