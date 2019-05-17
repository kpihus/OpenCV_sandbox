import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from math import log
from shapedetector import ShapeDetector
from helpers import Helpers
from shape import Line

import time

# https://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html
# https://stackoverflow.com/questions/15780210/python-opencv-detect-parallel-lines
# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python


cap = cv2.VideoCapture(0)

# Get current width of frame
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
# Get current height of frame
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(width), int(height)))

while cap.isOpened():
    start_time = time.time()
    ret, img = cap.read()
    # img = cv2.imread('cross.jpg')
    # img = cv2.imread('IMG_6496.JPG.JPG')
    # img = cv2.imread('curved2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()
    h = Helpers()

    rows, cols = img.shape[:2]

    lines = []
    angle1 = []
    angle2 = []
    i = 0

    img_center = (int(cols / 2), int(rows / 2))

    cv2.circle(img, img_center, 7, (0, 0, 0), -1) # Image center point

    for c in cnts:
        shape = sd.detect(c)
        print(shape)
        if shape == "rectangle":
            cv2.drawContours(img, [c], -1, (0, 0, 255), 2)
        else:
            continue

        M = cv2.moments(c)  # https://www.youtube.com/watch?v=AAbUfZD_09s
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # print((i, cX, cY))


        cv2.circle(img, (cX, cY), 7, (255, 0, 0), -1)
        cv2.putText(img, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        i = i + 1


        # bound angeled rectangle around shape
        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # cv2.drawContours(img, [box], 0, (0, 255,0), 2)

        # Fit line trough the shape
          # img: width, height
        # https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#gaf849da1fdafa67ee84b1e9a23b93f91f
        # Output line parameters. In case of 2D fitting, it should be a vector of 4 elements (like Vec4f) - (vx, vy, x0, y0),
        # where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line.
        [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        point1 = (cols - 1, righty)
        point2 = (0, lefty)

        # cv2.line(img, point1, point2, (0, 255, 0), 2)
        newline = Line((cX, cY), point1, point2, img_center)

        lines.append(newline)
        # plt.plot([cols - 1, righty], [0, lefty], color='k', linestyle='-', linewidth=2)
        # break

    lines.sort(key=lambda l: l.angle)

    bestline = Line((int(cols / 2), int(rows / 2)), (0, int(rows / 2)), (cols, int(rows / 2)), img_center) # Initial best line is worst ever, 90 degrees with flight path

    # find all good lines (closest to Y axis)
    goodlines = []
    for l in lines:
        if l.angle < abs(bestline.angle):
            goodlines.append(l)

    if len(goodlines) < 3:
        continue

    del goodlines[0]
    del goodlines[-1]


    bestAngle = h.averageLineDeg(goodlines)


    # cv2.line(img, bestline.p1, bestline.p2, (255, 255, 0), 2)
    # img = h.rotateImage(img, bestAngle) # Rotate image to angle

    centerline = Line((int(cols / 2), int(rows / 2)), (int(cols / 2), 0), (int(cols / 2), rows), img_center)
    # cv2.line(img, centerline.p1, centerline.p2, (0, 0, 255), 2)

    closest_to_c = h.closest_point_to((int(cols / 2), int(rows / 2)), goodlines)

    if len(goodlines) < 3:
        continue

    goodlines.sort(key=lambda l: l.centerdistance)
    cv2.circle(img, goodlines[0].guidepoint, 7, (0, 0, 255), -1)  # Most center point
    cv2.circle(img, goodlines[1].guidepoint, 7, (0, 255, 0), -1)  # ... one above it
    cv2.circle(img, goodlines[2].guidepoint, 7, (0, 255, 0), -1)  # ... and one below

    flight_line = Line(goodlines[0].guidepoint, goodlines[2].guidepoint, goodlines[1].guidepoint, img_center)

    cross_line = flight_line.plot_point(img_center, int(flight_line.angle) + 90, 250)
    cv2.circle(img, cross_line[1], 4, (255, 0, 255), -1)  # cross line endpoint


    cv2.line(img, flight_line.p1, flight_line.p2, (0, 255, 0), 2) # Draw ideal flight line
    cv2.line(img, cross_line[0], cross_line[1], (0, 10, 0), 2) # Line 90deg to flitht line

    cross_point = Line.line_intersection((flight_line.p1, flight_line.p2), (cross_line[0], cross_line[1]))
    cv2.circle(img, cross_point, 4, (255, 0, 255), -1)  # cross line endpoint

    drift_from_ideal = h.distance(cross_point, img_center)




    cv2.putText(img, "Yaw drift: "+str(flight_line.angle) + " deg", (100, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)
    cv2.putText(img, "Roll drift: "+str(drift_from_ideal) + " px", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)

    end_time = time.time() - start_time
    cv2.putText(img, "Calc time: "+str(end_time) + " sec", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 100), 2)


    cv2.imshow('Image', img)
    # cv2.imshow('Image', thresh)
    cv2.imwrite('out.jpg', img)
    # plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
out.release()
cv2.destroyAllWindows()


# cv2.imshow('image', img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# edges = cv2.Canny(im_bw, 200, 200)
#
#
# plt.subplot(3, 1, 1)
# plt.imshow(im_bw, cmap='gray', interpolation='bicubic')
# plt.title('Original')
#
# plt.subplot(3, 1, 2)
# plt.imshow(edges, cmap='gray', interpolation='bicubic')
# plt.title('Edges')


# plt.subplot(3, 1, 3)
# indices = np.where(edges != [0])
# coordinates = zip(indices[0], indices[1])
# plt.scatter(*zip(*coordinates), 0.1)


# plt.plot([50, 100], [80, 100], 'c', linewidth=5)

# plt.show()

# cv2.imwrite('somename.jpg', img) #save image

# cap = cv2.VideoCapture(0)
# # fourcc = cv2.VideoWriter_fourcc(*'XVID')
# # out = cv2.VideoWriter('out.avi', fourcc, 20.0, (640, 480))
#
# while True:
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     edges = cv2.Canny(gray, 200, 200)
#
#     indices = np.where(edges != [0])
#     coordinates = zip(indices[0], indices[1])
#
#     testList2 = [(elem1, log(elem2)) for elem1, elem2 in coordinates]
#     zip(*testList2)
#     plt.scatter(*zip(*testList2), 0.1)
#
#     plt.show()
#
#     # out.write(frame)
#     # cv2.imshow('frame', frame)
#     # cv2.imshow('frame2', gray)
#     cv2.imshow('framee', edges)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# # out.release()
# cv2.destroyAllWindows()
