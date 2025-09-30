import cv2
import numpy as np

image = np.zeros((500, 400, 3), np.uint8)
cv2.imshow("image", image)

#image[:] = (245, 135, 66)
#rgb = bgr

# image[100:150, 200:250] = (245, 135, 66)

cv2.rectangle(image, (100, 100), (200, 200), (245, 135, 66), 1)

cv2.line(image, (100, 100), (200, 200), (245, 135, 66), 1)

print(image.shape)
cv2.line(image, (0, image.shape[0] // 2), (image.shape[1], image.shape[0] // 2), (245, 135, 66), 1)

cv2.line(image, (image.shape[1] // 2, 0), (image.shape[1] // 2, image.shape[0]), (245, 135, 66), 1)

cv2.circle(image, (200, 200), 30, (245, 135, 66), 2)

cv2.putText(image, "Hi is this crusty crab?", (10, 150), cv2.FONT_HERSHEY_PLAIN, 2, (245, 135, 66), 2)
cv2.putText(image, "No this is Partick", (30, 250), cv2.FONT_HERSHEY_PLAIN, 2, (245, 135, 66), 2)

cv2.imshow("image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()