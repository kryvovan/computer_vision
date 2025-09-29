import cv2
import numpy

image = cv2.imread('images/img2.jpg')

#image = cv2.resize(image, (400, 500))

image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

print(image.shape)

kernel = numpy.ones((3, 3), numpy.uint8)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 40, 40)
image = cv2.dilate(image, kernel, iterations = 1)
image = cv2.erode(image, kernel, iterations = 1)

image = cv2.imshow('Ivan', image)


image2 = cv2.imread('images/img3.jpg')

print(image2.shape)

kernell = numpy.ones((3, 3), numpy.uint8)

image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = cv2.Canny(image2, 125, 125)
image2 = cv2.dilate(image2, kernell, iterations = 1)
image2 = cv2.erode(image2, kernell, iterations = 1)


image2 = cv2.imshow('Email', image2)



cv2.waitKey(0)
cv2.destroyAllWindows()