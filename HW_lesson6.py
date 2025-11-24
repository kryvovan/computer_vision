import cv2
import numpy as np


img = cv2.imread("images\img_hw.png")
img = cv2.resize(img, (1000, 600))
img_copy = img.copy()


img = cv2.GaussianBlur(img, (3, 3), 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower = np.array([0, 36, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)
img = cv2.bitwise_and(img, img, mask=mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 10:
        perimeter = cv2.arcLength(cnt, True)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])


        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = round(w / h, 2)
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2)

        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Square"
        elif len(approx) > 10:
            shape = "Oval"
        elif len(approx) > 5:
            shape = "Star"
        else:
            shape = "Other"

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(img_copy, (cx, cy), 4, (0, 255, 0), 1)
        cv2.putText(img_copy, f'Shape: {shape}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(img_copy, f'Area: {area}, P: {int(perimeter)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_copy, f'Aspect ratio: {aspect_ratio}, compactness: {compactness}', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)




cv2.imshow("image1", img_copy)
cv2.imshow("image2", img)


cv2.waitKey(0)
cv2.destroyAllWindows()