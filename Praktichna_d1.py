import cv2     #почти работающая версия
import os
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time
import math

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Константы
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
TRACKER = "bytetrack.yaml"
USE_WEBCAM = False  # False - для YouTube стрима
YOUTUBE_STREAM_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=xSDu_5Z82CPFtA8N"

# Классы транспортных средств (COCO dataset)
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

# ------------------- НАСТРОЙКА ЗОНЫ ПОДСЧЁТА -------------------
# Тут ты должен кликнуть на видео и выбрать 4 точки для двух линий
# Формат: [x1, y1, x2, y2] для первой линии и [x1, y1, x2, y2] для второй
# Рекомендую сначала запустить программу без обработки, нажать 's' и кликнуть точки

# ПРИМЕРНЫЕ КООРДИНАТЫ (нужно будет подстроить под твоё видео)
LINE1_POINTS = [(100, 400), (500, 400)]  # Первая линия (левая)
LINE2_POINTS = [(100, 450), (500, 450)]  # Вторая линия (правая)

# Реальное расстояние между линиями в метрах (нужно измерить)
REAL_DISTANCE_METERS = 5.0
# --------------------------------------------------------------

# Глобальные переменные для сбора точек
points = []
current_line = 1
selecting_points = False


def mouse_callback(event, x, y, flags, param):
    """Callback для выбора точек мышкой"""
    global points, current_line, selecting_points

    if event == cv2.EVENT_LBUTTONDOWN and selecting_points:
        points.append((x, y))
        print(f"Точка {len(points)}: ({x}, {y}) для линии {current_line}")

        if len(points) == 2:
            if current_line == 1:
                global LINE1_POINTS
                LINE1_POINTS = points.copy()
                print(f"Линия 1 установлена: {LINE1_POINTS}")
            else:
                global LINE2_POINTS
                LINE2_POINTS = points.copy()
                print(f"Линия 2 установлена: {LINE2_POINTS}")

            points = []
            current_line += 1

            if current_line > 2:
                selecting_points = False
                print("Обе линии установлены! Нажмите любую клавишу для продолжения...")


class VehicleTracker:
    def __init__(self):
        self.total_vehicles = set()  # Уникальные ID машин
        self.vehicles_crossed = set()  # Машины, пересекшие обе линии
        self.vehicle_speeds = []  # Скорости машин
        self.crossing_times = {}  # Время пересечения для каждой машины

        # Словарь для хранения времени пересечения каждой линии
        self.line1_crossing = {}
        self.line2_crossing = {}

        self.avg_speed = 0

    def check_line_crossing(self, track_id, center_y, line_y, line_points, current_time):
        """Проверяет пересечение линии"""
        x1, y1 = line_points[0]
        x2, y2 = line_points[1]

        # Проверяем, находится ли центр в районе линии
        if abs(center_y - line_y) < 5:
            if track_id not in self.line1_crossing and line_y == LINE1_POINTS[0][1]:
                self.line1_crossing[track_id] = current_time
                print(f"Машина {track_id} пересекла линию 1")
                return "line1"
            elif track_id not in self.line2_crossing and line_y == LINE2_POINTS[0][1]:
                self.line2_crossing[track_id] = current_time
                print(f"Машина {track_id} пересекла линию 2")
                return "line2"
        return None

    def calculate_speed(self, track_id, current_time):
        """Вычисляет скорость машины"""
        if track_id in self.line1_crossing and track_id in self.line2_crossing:
            time_diff = abs(self.line2_crossing[track_id] - self.line1_crossing[track_id])

            if time_diff > 0 and track_id not in self.vehicles_crossed:
                # Скорость = расстояние / время (м/с) -> км/ч
                speed_kmh = (REAL_DISTANCE_METERS / time_diff) * 3.6

                self.vehicle_speeds.append(speed_kmh)
                self.vehicles_crossed.add(track_id)
                self.avg_speed = np.mean(self.vehicle_speeds) if self.vehicle_speeds else 0

                print(f"Машина {track_id}: скорость {speed_kmh:.1f} км/ч")
                return speed_kmh
        return None


def main():
    global selecting_points

    # Загружаем модель
    model = YOLO(MODEL_PATH)

    # Открываем видео поток
    # Для YouTube нужно использовать yt-dlp или стрим URL
    cap = cv2.VideoCapture(YOUTUBE_STREAM_URL)

    if not cap.isOpened():
        print("Ошибка: Не могу открыть видеопоток!")
        print("Пробуем альтернативный метод...")
        # Альтернативный метод - используем yt-dlp
        import subprocess
        yt_cmd = f"yt-dlp -g {YOUTUBE_STREAM_URL}"
        try:
            stream_url = subprocess.check_output(yt_cmd, shell=True).decode().strip().split('\n')[0]
            cap = cv2.VideoCapture(stream_url)
        except:
            print("Не удалось получить стрим. Используем вебкамеру для теста...")
            cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0:
        frame_width, frame_height = 1280, 720

    # Создаем окно и устанавливаем callback для мыши
    cv2.namedWindow('Traffic Analysis')
    cv2.setMouseCallback('Traffic Analysis', mouse_callback)

    # Инициализируем трекер
    tracker = VehicleTracker()

    # Настройка сохранения видео
    output_path = os.path.join(OUTPUT_DIR, 'traffic_analysis.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("\n=== ИНСТРУКЦИЯ ===")
    print("1. Нажмите 's' для выбора точек линий")
    print("2. Кликните 2 точки для первой линии")
    print("3. Кликните 2 точки для второй линии")
    print("4. Нажмите 'q' для выхода")
    print("==================\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Переподключение к стриму...")
            cap = cv2.VideoCapture(YOUTUBE_STREAM_URL)
            continue

        # Рисуем линии
        cv2.line(frame, LINE1_POINTS[0], LINE1_POINTS[1], (0, 255, 0), 2)
        cv2.line(frame, LINE2_POINTS[0], LINE2_POINTS[1], (0, 255, 0), 2)

        # Добавляем метки для линий
        cv2.putText(frame, "Line 1", (LINE1_POINTS[0][0], LINE1_POINTS[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Line 2", (LINE2_POINTS[0][0], LINE2_POINTS[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Детекция и трекинг
        results = model.track(frame, conf=CONF_THRESH, tracker=TRACKER,
                              persist=True, verbose=False, classes=[2, 3, 5, 7])  # 2=car,3=motorcycle,5=bus,7=truck

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            track_ids = boxes.id.cpu().numpy()
            cls = boxes.cls.cpu().numpy()

            for i, (box, track_id, class_id) in enumerate(zip(xyxy, track_ids, cls)):
                x1, y1, x2, y2 = box.astype(int)
                tid = int(track_id)

                # Фильтруем только транспорт
                class_name = model.names[int(class_id)]
                if class_name not in VEHICLE_CLASSES:
                    continue

                # Центр объекта
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                current_time = time.time()

                # Проверяем пересечение линий
                line1_y = LINE1_POINTS[0][1]
                line2_y = LINE2_POINTS[0][1]

                if abs(center_y - line1_y) < 10:
                    tracker.check_line_crossing(tid, center_y, line1_y, LINE1_POINTS, current_time)
                if abs(center_y - line2_y) < 10:
                    tracker.check_line_crossing(tid, center_y, line2_y, LINE2_POINTS, current_time)

                # Вычисляем скорость
                speed = tracker.calculate_speed(tid, current_time)

                # Рисуем bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Подпись с ID и скоростью
                label = f'ID:{tid} {class_name}'
                if speed:
                    label += f' {speed:.1f}km/h'

                # Фон для текста
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (0, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Рисуем центр
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                tracker.total_vehicles.add(tid)

        # Отображаем статистику
        y_pos = 30
        cv2.putText(frame, f'Total vehicles crossed: {len(tracker.vehicles_crossed)}',
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 30
        cv2.putText(frame, f'Average speed: {tracker.avg_speed:.1f} km/h',
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 30
        cv2.putText(frame, f'Unique vehicles: {len(tracker.total_vehicles)}',
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 30
        cv2.putText(frame, f'Distance: {REAL_DISTANCE_METERS}m',
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Инструкция по выбору точек
        if selecting_points:
            cv2.putText(frame, f'Selecting line {current_line} - click 2 points',
                        (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Рисуем уже выбранные точки
            for pt in points:
                cv2.circle(frame, pt, 5, (0, 0, 255), -1)

        # Показываем кадр
        cv2.imshow('Traffic Analysis', frame)
        out.write(frame)

        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            selecting_points = True
            current_line = 1
            points = []
            print("Режим выбора точек активирован!")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Выводим итоговую статистику
    print("\n=== ИТОГОВАЯ СТАТИСТИКА ===")
    print(f"Всего машин пересекло переход: {len(tracker.vehicles_crossed)}")
    print(f"Средняя скорость: {tracker.avg_speed:.1f} км/ч")
    print(f"Всего уникальных машин: {len(tracker.total_vehicles)}")
    if tracker.vehicle_speeds:
        print(f"Минимальная скорость: {min(tracker.vehicle_speeds):.1f} км/ч")
        print(f"Максимальная скорость: {max(tracker.vehicle_speeds):.1f} км/ч")


if __name__ == "__main__":
    main()