import cv2
import numpy as np

image = cv2.imread('images/img2.jpg')



image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

print(image.shape)

cv2.rectangle(image, (150, 200), (330, 460), (245, 135, 66), 2)
cv2.putText(image, "Kryvobok Ivan", (150, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

cv2.imshow("Ivan", image)


cv2.waitKey(0)
cv2.destroyAllWindows()