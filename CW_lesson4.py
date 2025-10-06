import cv2
import numpy as np

img = cv2.imread('images/celovek-so-skresennymi-rukami.jpg')

scale = 10
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)
img_copy = img.copy()
img_copy_color = img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2)
img_copy = cv2.equalizeHist(img_copy) #посилення контрасту
img_copy = cv2.Canny(img_copy, 50, 150)

contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#RETR_EXT. - режим отримання контурів, знаходить крайній зовнішній контур. якщо обєкт має дирку, то вони будуть ігноруватися
#CHAIN_APPROX_SIMPL - процес наближеного вираження одних величин через інші

#малювання контурів, прямокутників та тексту

for cnt in contours:
    area = cv2.contourArea(cnt) #визначення площі контура
    if area > 50:
        x, y, w, h = cv2.boundingRect(cnt) #найменший прямокутник, в який вписаний контур

        #cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2) #(img, список контурів, -1 - всі контури з масиву, колір, товщина)

        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y - 5 > 10 else y + 15
        text = f'x: {x}, y: {y}, S:{int(area)}'
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


cv2.imshow('Borders and coordinates', img)
cv2.imshow('Copy border', img_copy_color)
cv2.waitKey(0)
cv2.destroyAllWindows()