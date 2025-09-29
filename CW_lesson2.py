import cv2
import numpy as np

# image = cv2.imread('images/img1.png')
# # image = cv2.resize(image, (400, 500))
# # image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
# # image = cv2.flip(image, 0)
# # image = cv2.GaussianBlur(image,(9,9),3 ) #рівень блюра лише непарні числа
#
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 130, 120)
# #1
# # image = cv2.dilate(image, None, iterations=1)
# kernel = np.ones((5, 5), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
#
# print(image.shape)
#
#
#
#
#
#
# cv2.imshow('Jesse', image)
# # cv2.imshow('image', image[0:200, 0:100])

#video
video = cv2.VideoCapture(0)
while True:
    mistake, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#cv2.waitKey(0)
cv2.destroyAllWindows()