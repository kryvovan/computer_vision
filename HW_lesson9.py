import cv2

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')

classes = []

with open("data/MobileNet/synset.txt", "r", encoding = "utf8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image = cv2.imread("images/cat.jpg")
image1 = cv2.imread("images/dog.jpg")
image2 = cv2.imread("images/chicken.jpg")
image3 = cv2.imread("images/banana.jpg")

images = [("cat.jpg", image), ("dog.jpg", image1), ("chicken.jpg", image2), ("banana.jpg", image3)]
results = []

def classify_image(name, img):

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (224, 224)),
        1.0 / 127.5,
        (224, 224),
        (127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()

    label = classes[idx] if idx < len(classes) else "unknown"
    confidence = float(preds[0][idx]) * 100

    print(f"{name}")
    print(f"{label}")
    print(f"{confidence:.2f}%")

    text = f'{label}: {int(confidence)}%'
    img_show = cv2.resize(img, (400, 300))
    cv2.putText(img_show, text, (10, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    cv2.imshow(name, img_show)

    results.append(label)
    return label

label1 = classify_image("cat.jpg", image)
label2 = classify_image("dog.jpg", image1)
label3 = classify_image("chicken.jpg", image2)
label4 = classify_image("banana.jpg", image3)


labels = [label1, label2, label3, label4]
unique_labels = set(labels)


for label in unique_labels:
    print(f"{label:30} | {labels.count(label)}")

cv2.waitKey(0)
cv2.destroyAllWindows()