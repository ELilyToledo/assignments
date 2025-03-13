import cv2
import numpy as np
import time
import roipoints as rp


'''This function will take the frame, or whatever other image is inputted, and pass it through
the various image processing filters in order to provide a clearer and defined image.
The filters it uses are greyscale, gaussian, canny edge detection, morphologyEx, thinning,
and dilate. It also puts the image through a contrast algorithm using split color channels.
'''
def applyfilters(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    #cv2.imshow('enhanced', enhanced)

    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    global edges
    edges = cv2.Canny(blurred, 70, 140)
    global closed
    #closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (np.ones((23, 23), np.uint8)))
    thinned = cv2.ximgproc.thinning(edges)
    global dilated
    dilated = cv2.dilate(thinned, np.ones((5, 5), np.uint8), iterations=1)

    #cv2.imshow("edge", edges)
    #cv2.imshow('dialated', dilated)
    #cv2.imshow("closed", closed)


'''This function takes the frame and the direction(found from the arrow detection function)
and then based on the direction it displays the corresponding transparent arrow PNG onto
the frame. It resizes the PNG and sets its coordinates in order to display the arrow.
'''
def displayarrow(frame, direction):
    # upload and set each transparent png to an arrow
    upar = cv2.imread('arrows/uparrow.png', cv2.IMREAD_UNCHANGED)
    rightar = cv2.imread('arrows/rightarrow.png', cv2.IMREAD_UNCHANGED)
    leftar = cv2.imread('arrows/leftarrow.png', cv2.IMREAD_UNCHANGED)

    if direction == 'right':
        arrow = rightar
    elif direction == 'left':
        arrow = leftar
    else:
        arrow = upar

    # resize the png
    h, w, _ = arrow.shape
    arrow = cv2.resize(arrow, (int(h * 0.2), int(w * 0.2)))
    arrbgr = arrow[:, :, :3]
    arralph = arrow[:, :, 3]

    # set the coordinates of where the arrow will be
    roix, roiy = 750, 35
    h, w = arrbgr.shape[:2]

    if roiy + h <= frame.shape[0] and roix + w <= frame.shape[1]:
        roi = frame[roiy:roiy + h, roix:roix + w]
        mask = arralph.astype(float) / 255.0

    for x in range(3):
        roi[:, :, x] = (mask * arrbgr[:, :, x] + (1 - mask) * roi[:, :, x])

    return frame


''' This function takes the frame and the road arrow templates and passes it through the
opencv template matching algorithm in order to detect when the car is turning. It does this
by passing the templates and the frame through the same filters and using template matching
to try and detect if an arrow can be found. Based on what it detects, the function will then
return a direction to be passed to the displayarrow function in order to display the corresponding
arrow onto the frame.
'''
def detectarrow(frame):
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Load the templates and convert to grayscale
    rightarrtemp = cv2.imread('rightarrowtemp.png', cv2.IMREAD_GRAYSCALE)
    leftarrtemp = cv2.imread('leftarrowtemp.png', cv2.IMREAD_GRAYSCALE)

    # Apply filters to the frame
    blurframe = cv2.GaussianBlur(grayframe, (5, 5), 0)
    edgeframe = cv2.Canny(blurframe, 70, 140)

    rightarrtemp = cv2.Canny(rightarrtemp, 70, 140)
    leftarrtemp = cv2.Canny(leftarrtemp, 70, 140)

    threshold = 0.7  

    # Perform template matching for left arrow first
    resultleft = cv2.matchTemplate(edgeframe, leftarrtemp, cv2.TM_CCOEFF_NORMED)
    locleft = np.where(resultleft >= threshold)
    
    if len(locleft[0]) > 0:  # If matches are found for the left arrow
        for pt in zip(*locleft[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + leftarrtemp.shape[1], pt[1] + leftarrtemp.shape[0]), (0, 0, 255), 2)
        return 'left'
    
    # Perform template matching for right arrow if left arrow was not found
    resultright = cv2.matchTemplate(edgeframe, rightarrtemp, cv2.TM_CCOEFF_NORMED)
    locright = np.where(resultright >= threshold)
    
    if len(locright[0]) > 0:  # If matches are found for the right arrow
        for pt in zip(*locright[::-1]):
            cv2.rectangle(frame, pt, (pt[0] + rightarrtemp.shape[1], pt[1] + rightarrtemp.shape[0]), (0, 0, 255), 2)
        return 'right'

    # If no arrows detected, return 'up' (default)
    return 'up'

''' This function detects the lane lines and calculates a centerline, and then overlays them.
It does this by first, defining the roi points and drawing it out. Then it applies a mask to 
the frame in the shape of the roi. Then, it passes the roi through the filters, and then
uses the outputted image to detect the lanes using Hough Lines P. It then calculates the slope
for each line detected and filters the lines baed on their slope into leftlines or rightlines.
Then it extends the lines and takes the average of all the lines in both lists in order to
find the lane lines. It then calculates the centerline by calculating all the midpoints between
the two lines.
'''
def overlay(frame):
    elapst = time.time() - start
    height, width = frame.shape[:2]

    roipts = np.array([
        [0.1 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.55 * height],  # left midpoint
        [0.55 * width, 0.55 * height],  # right midpoint
        [0.9 * width, height * 0.9]  # bottom right
    ], np.int32)

    cv2.polylines(frame, [roipts], isClosed=True, color=(0, 0, 0), thickness=1)

    roipts = rp.points(elapst, width, height)

    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roipts], (255, 255, 255))

    roi = cv2.bitwise_and(frame, mask)

    ogroi = roi

    applyfilters(roi)

    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=55)

    if lines is not None:

        leftlanes = []
        rightlanes = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:
                slope = (y2 - y1) / (x2 - x1)
            else:
                float('inf')

            if abs(slope) > 0.5:
                if slope < 0:
                    leftlanes.append(line[0])
                else:
                    rightlanes.append(line[0])

        y_bottom = int(height * 0.9)
        y_top = int(height * 0.55)

        def extend_line(line, y1, y2):
            x1, y1_, x2, y2_ = line
            if x1 == x2:
                return [(x1, y1), (x1, y2)]
            else:
                slope = (y2_ - y1_) / (x2 - x1)
                intercept = y1_ - slope * x1
                if slope != 0:
                    x1_new = int((y1 - intercept) / slope)
                    x2_new = int((y2 - intercept) / slope)
                    return [(x1_new, y1), (x2_new, y2)]
                else:
                    return [(x1, y1), (x2, y2)]

        leftlanes = [extend_line(line, y_top, y_bottom) for line in leftlanes]
        rightlanes = [extend_line(line, y_top, y_bottom) for line in rightlanes]

        def average_lines(lines):
            if len(lines) == 0:
                return None
            x1_avg = int(np.mean([line[0][0] for line in lines]))
            x2_avg = int(np.mean([line[1][0] for line in lines]))
            return [(x1_avg, y_top), (x2_avg, y_bottom)]

        leftlane = average_lines(leftlanes)
        rightlane = average_lines(rightlanes)

        if leftlane is not None:
            cv2.line(frame, leftlane[0], leftlane[1], (0, 0, 255), 4)
        if rightlane is not None:
            cv2.line(frame, rightlane[0], rightlane[1], (0, 0, 255), 4)

        centerline = []
        # for each corresponding point in the two polylines, calculate the midpoint for the centerline
        if leftlane is not None and rightlane is not None:
            for p1, p2 in zip(leftlane, rightlane):
                midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                centerline.append(midpoint)

            if len(centerline) > 0:
                # made back into a numpy array in order to draw them as a polyline
                centerline = np.array(centerline)
                cv2.polylines(frame, [centerline], isClosed=False, color=(255, 0, 255), thickness=4,
                              lineType=cv2.LINE_AA)

    direction = detectarrow(ogroi)
    frame = displayarrow(frame, direction)

    return frame


def proccessframe(frame):
    frame = cv2.resize(frame, (900, 600))
    frame = overlay(frame)
    return frame

start = time.time()

#in order for the video not to be played when running the GUI
if __name__ == "__main__":
    cap = cv2.VideoCapture("roadvid2.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break
        
        frame = proccessframe(frame)

        cv2.imshow("Curved Path Overlay", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

