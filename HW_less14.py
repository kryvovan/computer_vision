import cv2
import shutil
import os

PROJECT_DIR = os.path.dirname(__file__)

IMAGES_DIR = os.path.join(PROJECT_DIR, "images")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

OUT_DIR = os.path.join(PROJECT_DIR, "output")
PEOPLE_DIR = os.path.join(OUT_DIR, "people")
NO_PEOPLE_DIR = os.path.join(OUT_DIR, "no_people")

os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

print("PROJECT_DIR:", PROJECT_DIR)
print("IMAGES_DIR:", IMAGES_DIR)
print("OUT_DIR:", OUT_DIR)
print("PEOPLE_DIR:", PEOPLE_DIR)
print("NO_PEOPLE_DIR:", NO_PEOPLE_DIR)

PROTOTXT_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.prototxt.txt")
MODEL_PATH = os.path.join(MODELS_DIR, "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

PERSON_CLASS_ID = CLASSES.index("person")
CONF_THRESHOLD = 0.6


def detect_people(image):

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image, 0.007843, size=(300, 300), mean=(127.5, 127.5, 127.5)
    )

    net.setInput(blob)
    detections = net.forward()

    boxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = detections[0, 0, i, 1]

        if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
            box = detections[0, 0, i, 3:7]
            x1 = int(box[0] * w)
            y1 = int(box[1] * h)
            x2 = int(box[2] * w)
            y2 = int(box[3] * h)


            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            boxes.append((x1, y1, x2, y2))
            confidences.append(confidence)

    return boxes, confidences


allowed_ext = (".jpg", ".png", ".jpeg", ".bmp")
files = os.listdir(IMAGES_DIR)

count_people = 0
count_no_people = 0

for file in files:
    if not file.lower().endswith(allowed_ext):
        continue

    in_path = os.path.join(IMAGES_DIR, file)
    img = cv2.imread(in_path)

    if img is None:
        print("Не можу прочитати файл:", in_path)
        continue

    boxes, confidences = detect_people(img)
    people_count = len(boxes)

    if people_count == 0:
        out_folder = NO_PEOPLE_DIR
        count_no_people += 1
    else:
        out_folder = PEOPLE_DIR
        count_people += 1


    shutil.copyfile(in_path, os.path.join(out_folder, file))
    boxed = img.copy()

    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(boxed,f"{conf:.2f}",(x1, max(0, y1 - 7)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),2)

    cv2.putText(boxed,f"People count: {people_count}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 255, 0),2)

    boxed_path = os.path.join(out_folder, "boxed_" + file)
    cv2.imwrite(boxed_path, boxed)

    print(f"{file}: People count = {people_count} -> {out_folder}")

print("Images with people:", count_people)
print("Images with NO people:", count_no_people)