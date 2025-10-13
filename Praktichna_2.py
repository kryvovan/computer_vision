import cv2
import numpy as np

img = cv2.imread('images/shapes2.jpg')
img = cv2.resize(img, (500, 600))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img = cv2.GaussianBlur(img, (5, 5), 2)

lowerred = np.array([167, 0, 0])
upperred = np.array([179, 255, 255])
maskred = cv2.inRange(img, lowerred, upperred)

lowerblue = np.array([71, 87, 0])
upperblue = np.array([118, 255, 255])
maskblue = cv2.inRange(img, lowerblue, upperblue)

lowergreen = np.array([39, 0, 0])
uppergreen = np.array([51, 255, 255])
maskgreen = cv2.inRange(img, lowergreen, uppergreen)

loweryellow = np.array([5, 56, 113])
upperyellow = np.array([32, 255, 232])
maskyellow = cv2.inRange(img, loweryellow, upperyellow)

mask_total = cv2.bitwise_or(maskred, maskblue)
mask_total = cv2.bitwise_or(mask_total, maskgreen)
mask_total = cv2.bitwise_or(mask_total, maskyellow)

img = cv2.bitwise_or(img_copy, img_copy, mask = mask_total)

countors, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in countors:
    area = cv2.contourArea(cnt)
    if area > 200:
        x, y, w, h = cv2.boundingRect(cnt)
        perimeter = cv2.arcLength(cnt, True) #обчислюється периметр контура
        M = cv2.moments(cnt) #моменти контуру
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            shape = "SQUARE"
        elif len(approx) == 3:
            shape = "TRIANGLE"
        elif len(approx) > 8:
            shape = "OVAL"
        else:
            shape = "inshe"

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0),
                         2)  # (img, список контурів, -1 - всі контури з масиву, колір, товщина)
        hs = 0
        if hs >= 0 and h <=10 and h >=160 and h <=179:
            color = "red"
        elif hs >=26 and hs <= 35:
            color = "yellow"
        elif hs >=36 and hs <= 85:
            color = "green"
        elif hs >= 101 and hs <= 130:
            color = "blue"



        cv2.putText(img_copy, f'S: {int(area)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(img_copy, f'Shape: {shape}', (x, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        #cv2.putText(img_copy, f'Color: {color}', (x, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)



cv2.imshow('mask', img)
cv2.imshow('shapes', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()