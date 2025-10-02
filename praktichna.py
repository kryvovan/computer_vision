import cv2
import numpy as np
image = np.full((400, 600, 3), (194, 194, 194), np.uint8)

photo = cv2.imread("images/img2.jpg")
photo = cv2.resize(photo, (photo.shape[0] // 8, photo.shape[1] // 6))

x, y = 30, 30
h, w = photo.shape[:2]
image[y:y+h, x:x+w] = photo


qr = cv2.imread("images/qr.jpg")
qr = cv2.resize(qr, (100, 100))

image[230:330, 450:550] = qr

print(photo.shape)


cv2.rectangle(image, (10, 10), (590, 390), (112, 112, 112), 3)

cv2.putText(image, "Ivan Kryvobok", (200, 100), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 0), 2)
cv2.putText(image, "PL NTUU 'KPI' Student", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (77, 77, 77), 2)
cv2.putText(image, "Email: kryvobokivan0707@gmail.com", (200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (53, 53, 140), 1)
cv2.putText(image, "Phone: +380504740880", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (53, 53, 140), 1)
cv2.putText(image, "07/07/2010", (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (53, 53, 140), 1)
cv2.putText(image, "OpenCV Business Card", (150, 370), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)


cv2.imshow("business_card", image)

cv2.imwrite("business_card.png", image)

cv2.waitKey(0)
cv2.destroyAllWindows()