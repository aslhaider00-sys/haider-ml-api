import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)

offset = 20
imgSize = 300

folder = "Data/Z"
counter = 0

if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    key = cv2.waitKey(1)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        try:
            imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)

                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)

                imgWhite[:, wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)

                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)

                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("imgCrop", imgCrop)
            cv2.imshow("imgWhite", imgWhite)

            if key == ord("s"):
                counter += 1
                filename = f"{folder}/Image_{time.time()}.png"

                saved = cv2.imwrite(filename, imgWhite)

                if saved:
                    print("Saved:", filename)
                else:
                    print("Failed to save!")

        except:
            pass

    cv2.imshow("Image", img)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



