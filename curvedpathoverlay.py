# Import the neccesary libraries (opencv and numpy)

# Define the function curvedetection(frame, roi)
    # Process the frame with multiple filters (grayscale, blurrring, edge detection)
    # Create a mask with the parameters of the set ROI
    # Use the function findContours to detect the contours inside the mask
    # FOR each contour in the contour array,
        # IF its arclength is greater than 100 THEN add it to a the filtered contours list
    # IF the filtered contours list has 2 or more contours in it THEN
        # SORT the filter contours list by area, from least to greatest
        # SET cont1 and cont2 to the first two elements in the list
        # SET poly1 and poly2 as polylines estimated to the 8th degree using cont1 and cont2
        # Shorten poly1 and pol2 in order to limit wrap arounds
        # Draw poly1 and poly2 as polylines
        # FOR each point in poly1 and poly2
        # CALCULATE the midpoint and append it to the list centerline
        # SET centerline as a numpy array
        # IF there are values in centerline 
        # THEN approximate it to the 8th degree and draw it as a polyline
    # Draw the roi rectangle on the frame

# CALL the video stream from the camera
# SET the roi to the wanted values
# WHILE the camera can be accessed
    # IF the stream cannot be processed, 
        #END
    # CALL function curvedetection and SET it to the variable overlay
    # DISPLAY the overlay image stream
    # IF q is clicked THEN END

import cv2
import numpy as np


def curvedetection(frame, roi):
    # process the image and apply filters
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    edges = cv2.Canny(blurred, 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (np.ones((43, 43), np.uint8)))

    #cv2.imshow("edge", edges)
    #cv2.imshow("closed", closed)

    # call roi parameters and set them to the value of the mask
    rx, ry, rw, rh = roi
    mask = np.zeros_like(closed)
    mask[ry:ry + rh, rx:rx + rw] = closed[ry:ry + rh, rx:rx + rw]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filconts = []

    # filter the contours by arc length in order to ignore the unimportant shorter ones
    for contour in contours:
        if cv2.arcLength(contour, True) > 100:
            filconts.append(contour)

    if len(filconts) >= 2:
        # in order to use the two largest contours
        filconts = sorted(filconts, key=cv2.contourArea, reverse=True)
        cont1 = filconts[0]
        cont2 = filconts[1]

        # approximating it to the 8th degree because an 8th degree polynomial is most like the paths we are dealing with
        poly1 = cv2.approxPolyDP(cont1, 8, closed=True)
        poly2 = cv2.approxPolyDP(cont2, 8, closed=True)

        # in order to make sure we are cutting the right contours by the right amount; ensures reliablity when rotating the path
        if len(poly1) >= len(poly2):
            poly1 = poly1[:int(len(poly1) * 0.6)]
            poly2 = poly2[:int(len(poly2) * 0.6)]
            #print("poly1 longer")
        else:
            poly1 = poly1[:int(len(poly1) * 0.6)]
            poly2 = poly2[:int(len(poly2) * 0.5)]
            #print("poly2 longer")

        cv2.polylines(frame, [poly1], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [poly2], isClosed=False, color=(0, 255, 0), thickness=4, lineType=cv2.LINE_AA)

        centerline = []
        # for each corresponding point in the two polylines, calculate the mipoint for the centerline
        for p1, p2 in zip(poly1, poly2):
            midpoint = ((p1[0][0] + p2[0][0]) // 2, (p1[0][1] + p2[0][1]) // 2)
            centerline.append(midpoint)

        if len(centerline) > 0:
            # made back into a numpy array in order to approximate the points to an 8th degree polynomial and draw them as a polyline
            centerline = np.array(centerline)
            centerline = cv2.approxPolyDP(centerline, epsilon=8, closed=True)
            cv2.polylines(frame, [centerline], isClosed=False, color=(255, 0, 255), thickness=4, lineType=cv2.LINE_AA)

    # drawing a rectangle on the boundaries of the roi
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    return frame


cap = cv2.VideoCapture(0)

# setting the correct values for the roi and how it will appear on the stream
roi = (200, 100, 900, 800)
#roi = (250, 50, 800, 600)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    overlay = curvedetection(frame, roi)
    cv2.imshow("Curved Path Overlay", overlay)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
