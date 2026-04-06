import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Start ng camera
cap = cv2.VideoCapture(0)

# Setup Detector at Classifier
detector = HandDetector(maxHands=1)
classifier = Classifier("myproject/Model/keras_model.h5", "myproject/Model/labels.txt")

# Configuration (Manual type to avoid U+00A0 error)
offset = 20
imgSize = 300

# Kumpletong Labels A-Z (26) at 1-10 (10) = 36 classes total
labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
]

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        
        # Boundary calculation
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Prediction logic
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            
            # Index check para iwas "Index out of range"
            if index < len(labels):
                current_label = labels[index]
                cv2.rectangle(imgOutput, (x - offset, y - offset - 60),
                            (x + w + offset, y - offset), (255, 255, 255), cv2.FILLED)
                cv2.putText(imgOutput, current_label, (x, y - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                            (x + w + offset, y + h + offset), (0, 0, 255), 4)

    cv2.imshow("Image", imgOutput)
    
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()