import cv2
import numpy as np


def nothing(x):
    pass


def initialize_trackbars(intialTracbarVals=0):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 120)
    cv2.createTrackbar("Threshold1", "Trackbars", 50, 255, nothing)
    cv2.createTrackbar("Threshold2", "Trackbars", 50, 255, nothing)


def get_trackbars_val():
    threshold1 = cv2.getTrackbarPos("Threshold1", "Trackbars")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Trackbars")
    return threshold1, threshold2


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def stack_images(img_arr, scale=1, labels=[]):

    rows = len(img_arr)
    cols = len(img_arr[0])

    width = img_arr[0][0].shape[1]
    height = img_arr[0][0].shape[0]

    for i in range(rows):
        for j in range(cols):
            img_arr[i][j] = cv2.resize(img_arr[i][j], (0, 0), None, scale, scale)
            if len(img_arr[i][j].shape) == 2:
                img_arr[i][j] = cv2.cvtColor(img_arr[i][j], cv2.COLOR_GRAY2BGR)

    image_blank = np.zeros((height, width, 3), np.uint8)

    horz = [image_blank for _ in range(rows)]

    for i in range(rows):
        horz[i] = np.hstack(img_arr[i])
    ver = np.vstack(horz)

    if len(labels) != 0:
        width = int(ver.shape[1] / cols)
        height = int(ver.shape[0] / rows)

        for i in range(rows):
            for j in range(cols):
                cv2.rectangle(
                    ver,
                    (width * j, height * i),
                    (
                        width * j + len(labels[i][j]) * 13 + 27,
                        height * i + 30,
                    ),
                    (255, 255, 255),
                    cv2.FILLED,
                )
                cv2.putText(
                    ver,
                    labels[i][j],
                    (width * j + 10, height * i + 20),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )
    return ver


def reorder(points):

    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)

    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]

    return new_points


def drawRectangle(img, biggest, thickness):
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[0][0][0], biggest[0][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[2][0][0], biggest[2][0][1]),
        (0, 255, 0),
        thickness,
    )
    cv2.line(
        img,
        (biggest[3][0][0], biggest[3][0][1]),
        (biggest[1][0][0], biggest[1][0][1]),
        (0, 255, 0),
        thickness,
    )

    return img
