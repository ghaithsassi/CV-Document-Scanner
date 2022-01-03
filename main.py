import cv2
import numpy as np
import os
import time

import requests
import imutils

import utils

#########################################
URL = "http://192.168.0.24:8080/shot.jpg"

#########################################
CAMPORT = 0
heightImg = 360
widthImg = 640
#########################################


# --------------------------------------#
def main():
    # cap = cv2.VideoCapture(CAMPORT, cv2.CAP_DSHOW)
    # cap.set(10, 160)

    utils.initialize_trackbars()

    while True:
        # success, img = cap.read()
        img_resp = requests.get(URL)
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        img = imutils.resize(img, width=widthImg, height=heightImg)

        success = True
        if success:
            # img = cv2.resize(img, (heightImg, widthImg))  # RESIZE IMAGE
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CONVERT IMAGE TO GRAY SCALE
            gblur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
            thr1, thr2 = utils.get_trackbars_val()
            imgthreshold = cv2.Canny(gblur, threshold1=thr1, threshold2=thr2)
            kernel = np.ones((5, 5))

            img_dial = cv2.dilate(imgthreshold, kernel, iterations=2)  # APPLY DILATION
            imgthreshold = cv2.erode(img_dial, kernel, iterations=1)  # APPLY EROSION

            ## FIND ALL CONTOURS
            img_contours = img.copy()
            contours, hierarchy = cv2.findContours(
                imgthreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                img_contours, contours, -1, (0, 255, 0), 10
            )  # DRAW ALL DETECTED CONTOURS

            # FIND THE BIGGEST CONTOUR
            img_biggest_contour = img.copy()
            biggest, maxArea = utils.biggestContour(contours)

            # CREATE A BLANK IMAGE
            img_adaptive_thr = img_warp = np.zeros((heightImg, widthImg, 3), np.uint8)

            if biggest.size != 0:
                biggest = utils.reorder(biggest)
                cv2.drawContours(
                    img_biggest_contour, biggest, -1, (0, 255, 0), 20
                )  # DRAW THE BIGGEST CONTOUR
                img_biggest_contour = utils.drawRectangle(
                    img_biggest_contour, biggest, 2
                )

                # PREPARE POINTS FOR WARP
                pts1 = np.float32(biggest)
                pts2 = np.float32(
                    [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
                )
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                img_warp = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

                # REMOVE 10 PIXELS FORM EACH SIDE
                img_warp = img_warp[
                    10 : img_warp.shape[0] - 10, 10 : img_warp.shape[1] - 10
                ]
                img_warp = cv2.resize(img_warp, (widthImg, heightImg))

                # APPLY ADAPTIVE THRESHOLD
                img_warp_gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
                # img_adaptive_thr = cv2.adaptiveThreshold(img_warp_gray, 255, 1, 1, 7, 2)
                # img_adaptive_thr = cv2.bitwise_not(img_adaptive_thr)
                # img_adaptive_thr = cv2.medianBlur(img_adaptive_thr, 3)

            img_arr = [
                [img, imgthreshold, img_contours],
                [img_biggest_contour, img_warp, img_warp_gray],
            ]
            labels = [
                ["Original", "Threshold", "Contours"],
                [" Bigest Contour", "Document", "Document gary"],
            ]
            results = utils.stack_images(img_arr, scale=1, labels=labels)

            cv2.imshow("window", results)
            waitkey = cv2.waitKey(1) & 0xFF
            if waitkey == ord("q"):
                break
            elif waitkey == ord("s"):
                name = "scan_" + str(time.time()) + ".jpg"
                image_name = os.path.join("saves", name)
                print(image_name)
                cv2.imwrite(image_name, img_adaptive_thr)

    # cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
