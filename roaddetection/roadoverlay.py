import cv2
import numpy as np


def displayarrow(frame):
    # upload and set each transparent png to an arrow
    upar = cv2.imread('uparrow.png', cv2.IMREAD_UNCHANGED)
    #rightar = cv2.imread('arrows/rightarrow.png', cv2.IMREAD_UNCHANGED)
    #leftar = cv2.imread('arrows/leftarrow.png', cv2.IMREAD_UNCHANGED)

    #resize the png
    arrow = upar
    h, w, _ = arrow.shape
    arrow = cv2.resize(arrow, (int(h*0.2), int(w*0.2)))

    arrbgr = arrow[:, :, :3]
    arralph = arrow[:, :, 3]

    #set the coordinates of where the arrow will be
    roix, roiy = 750, 35
    h, w = arrbgr.shape[:2]

    if roiy + h <= frame.shape[0] and roix + w <= frame.shape[1]:
        roi = frame[roiy:roiy + h, roix:roix + w]
        mask = arralph.astype(float) / 255.0

    for x in range(3):
        roi[:, :, x] = (mask * arrbgr[:, :, x] + (1 - mask) * roi[:, :, x])

    return frame


def overlay(frame):
    height, width = frame.shape[:2]

    roipts = np.array([
        [0.1 * width, height * 0.9],  # bottom left
        [0.45 * width, 0.55 * height],  # left midpoint
        [0.55 * width, 0.55 * height],  # right midpoint
        [0.9 * width, height * 0.9]  # bottom right
    ], np.int32)

    cv2.polylines(frame, [roipts], isClosed=True, color=(0, 0, 255), thickness=1)
    
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [roipts], (255, 255, 255))

    roi = cv2.bitwise_and(frame, mask)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 90)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, (np.ones((21, 21), np.uint8)))

    cv2.imshow("edge", edges)
    cv2.imshow("closed", closed)

    lines = cv2.HoughLinesP(closed, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=55)

    if lines is not None:

        leftlanes = []
        rightlanes = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf') 
            
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

        left_lane = average_lines(leftlanes)
        right_lane = average_lines(rightlanes)

        if left_lane is not None:
            cv2.line(frame, left_lane[0], left_lane[1], (0, 0, 255), 4)
        if right_lane is not None:
            cv2.line(frame, right_lane[0], right_lane[1], (0, 0, 255), 4)

        centerline = []
        # for each corresponding point in the two polylines, calculate the mipoint for the centerline
        for p1, p2 in zip(left_lane, right_lane):
            midpoint = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            centerline.append(midpoint)

        if len(centerline) > 0:
            # made back into a numpy array in order to draw them as a polyline
            centerline = np.array(centerline)
            cv2.polylines(frame, [centerline], isClosed=False, color=(255, 0, 255), thickness=4, lineType=cv2.LINE_AA)
         
    frame = displayarrow(frame)

    return frame


cap = cv2.VideoCapture("roadvid.mov")


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break
    frame = cv2.resize(frame, (900, 700))

    overlay(frame)

    cv2.imshow("Curved Path Overlay", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
