import cv2
import os
from ultralytics import YOLO


PROJECT_DIR = os.path.dirname(__file__)
VIDEO_PATH = os.path.join(PROJECT_DIR, 'video.mp4')

CONF_THRESH = 0.5

VEHICLE_CLASSES = [2, 3, 5, 7, 8, 9]

CLASS_NAMES = {
    2: 'car',
    3: 'bike',
    5: 'bus',
    7: 'truck'
}

CLASS_COLORS = {
    'car': (0, 255, 0),
    'bike': (0, 255, 255),
    'bus': (0, 0, 255),
    'truck': (255, 0, 255)
}

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(VIDEO_PATH)


frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
output_path = os.path.join(PROJECT_DIR, 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame, conf=CONF_THRESH, classes=VEHICLE_CLASSES)

    current_counts = {name: 0 for name in ['car', 'bike', 'bus', 'truck']}

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())

            class_name = CLASS_NAMES.get(class_id, 'авто')
            color = CLASS_COLORS.get(class_name, (0, 255, 0))

            current_counts[class_name] += 1

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f'{class_name}'
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    y_pos = 30
    cv2.putText(frame, 'Transport na kadri:', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for vehicle, count in current_counts.items():
        y_pos += 25
        color = CLASS_COLORS.get(vehicle, (255, 255, 255))
        cv2.putText(frame, f'{vehicle}: {count}', (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    writer.write(frame)

    cv2.imshow('zalik', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()