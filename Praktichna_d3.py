import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import csv
from datetime import datetime
import subprocess
import sys

PROJECT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.3
TRACKER = "bytetrack.yaml"
USE_WEBCAM = False
YOUTUBE_STREAM_URL = "https://www.youtube.com/live/Lxqcg1qt0XU?si=xSDu_5Z82CPFtA8N"


def get_youtube_stream_url(youtube_url):
    """Get direct stream URL from YouTube using yt-dlp"""
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except:
        print("yt-dlp not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)

    try:
        cmd = [
            'yt-dlp',
            '-g',
            '-f', 'best[height<=720]',
            youtube_url
        ]

        print("Getting stream URL...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stream_url = result.stdout.strip().split('\n')[0]

        if stream_url:
            print("Stream URL obtained")
            return stream_url
        else:
            print("Failed to get stream URL")
            return None

    except subprocess.CalledProcessError as e:
        print(f"yt-dlp error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


# Vehicle classes and their colors
VEHICLE_CLASSES = {
    'car': (0, 255, 0),  # Green
    'motorcycle': (0, 165, 255),  # Orange
    'bus': (0, 0, 255),  # Red
    'truck': (255, 0, 0),  # Blue
    'bicycle': (255, 255, 0)  # Cyan
}

# ------------------- CROSSING ZONE SETUP -------------------
LINE1_POINTS = [(200, 300), (600, 300)]
LINE2_POINTS = [(200, 350), (600, 350)]
REAL_DISTANCE_METERS = 5.0
# --------------------------------------------------------------

points = []
current_line = 1
selecting_points = False


def mouse_callback(event, x, y, flags, param):
    global points, current_line, selecting_points, LINE1_POINTS, LINE2_POINTS

    if event == cv2.EVENT_LBUTTONDOWN and selecting_points:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y}) for line {current_line}")

        if len(points) == 2:
            if current_line == 1:
                LINE1_POINTS = points.copy()
                print(f"Line 1 set: {LINE1_POINTS}")
            else:
                LINE2_POINTS = points.copy()
                print(f"Line 2 set: {LINE2_POINTS}")

            points = []
            current_line += 1

            if current_line > 2:
                selecting_points = False
                print("Both lines set!")


class VehicleTracker:
    def __init__(self):
        self.total_vehicles = set()
        self.vehicles_crossed = set()
        self.vehicle_speeds = []
        self.vehicle_current_speed = {}
        self.vehicle_class = {}
        self.line1_crossing = {}
        self.line2_crossing = {}
        self.avg_speed = 0
        self.crossed_vehicles_data = []

    def check_line_crossing(self, track_id, center_y, current_time, class_name):
        if track_id not in self.vehicle_class:
            self.vehicle_class[track_id] = class_name

        # Check line 1
        line1_y = LINE1_POINTS[0][1]
        if abs(center_y - line1_y) < 15:
            if track_id not in self.line1_crossing:
                self.line1_crossing[track_id] = current_time
                print(f"{class_name} {track_id} crossed line 1")
                return "line1"

        # Check line 2
        line2_y = LINE2_POINTS[0][1]
        if abs(center_y - line2_y) < 15:
            if track_id not in self.line2_crossing:
                self.line2_crossing[track_id] = current_time
                print(f"{class_name} {track_id} crossed line 2")
                return "line2"
        return None

    def calculate_speed(self, track_id, current_time, class_name):
        if track_id in self.line1_crossing and track_id in self.line2_crossing:
            time_diff = abs(self.line2_crossing[track_id] - self.line1_crossing[track_id])

            if time_diff > 0.1 and track_id not in self.vehicles_crossed:
                speed_kmh = (REAL_DISTANCE_METERS / time_diff) * 3.6

                if 3 < speed_kmh < 150:
                    self.vehicle_speeds.append(speed_kmh)
                    self.vehicle_current_speed[track_id] = speed_kmh
                    self.vehicles_crossed.add(track_id)
                    self.avg_speed = np.mean(self.vehicle_speeds) if self.vehicle_speeds else 0

                    self.crossed_vehicles_data.append({
                        'track_id': track_id,
                        'class': class_name,
                        'speed': round(speed_kmh, 1),
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'line1_time': self.line1_crossing[track_id],
                        'line2_time': self.line2_crossing[track_id]
                    })

                    print(f"{class_name} {track_id}: speed {speed_kmh:.1f} km/h")
                    return speed_kmh

        return self.vehicle_current_speed.get(track_id, None)


def draw_vehicle_box(frame, x1, y1, x2, y2, speed, class_name, track_id, color):
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Text
    if speed:
        main_label = f'{class_name} #{track_id}'
        speed_label = f'{speed:.1f} km/h'
        speed_color = (0, 0, 255)
    else:
        main_label = f'{class_name} #{track_id}'
        speed_label = 'waiting...'
        speed_color = (100, 100, 100)

    # Text sizes
    (w_main, h_main), _ = cv2.getTextSize(main_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (w_speed, h_speed), _ = cv2.getTextSize(speed_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

    # Background for main info
    cv2.rectangle(frame, (x1, y1 - h_main - 10), (x1 + w_main, y1), (50, 50, 50), -1)
    cv2.putText(frame, main_label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Background for speed
    if speed:
        cv2.rectangle(frame, (x1, y1 - h_main - h_speed - 25),
                      (x1 + w_speed, y1 - h_main - 15), speed_color, -1)
        cv2.putText(frame, speed_label, (x1 + 5, y1 - h_main - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (x1 + w_main + 10, y1 - h_speed - 10),
                      (x1 + w_main + w_speed + 20, y1), speed_color, -1)
        cv2.putText(frame, speed_label, (x1 + w_main + 15, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Center point
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)


def save_to_csv(tracker):
    csv_path = os.path.join(OUTPUT_DIR, f'vehicle_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['track_id', 'class', 'speed', 'time', 'line1_time', 'line2_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for data in tracker.crossed_vehicles_data:
            writer.writerow(data)

    print(f"Data saved to: {csv_path}")
    return csv_path


def main():
    global selecting_points, current_line, points

    print("Starting Traffic Analysis Program")
    print("=" * 50)

    # Load model
    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)
    print("Model loaded!")

    # Get video stream
    if USE_WEBCAM:
        print("Using webcam...")
        cap = cv2.VideoCapture(0)
    else:
        print("Getting YouTube stream with yt-dlp...")
        stream_url = get_youtube_stream_url(YOUTUBE_STREAM_URL)

        if stream_url:
            print(f"Connecting to stream...")
            cap = cv2.VideoCapture(stream_url)

            attempts = 0
            while not cap.isOpened() and attempts < 5:
                print(f"Connection attempt {attempts + 1}/5...")
                time.sleep(2)
                cap = cv2.VideoCapture(stream_url)
                attempts += 1
        else:
            print("Failed to get YouTube stream!")
            print("Tip: Try downloading video locally:")
            print(f"   yt-dlp -f best -o video.mp4 {YOUTUBE_STREAM_URL}")
            return

    if not cap.isOpened():
        print("Failed to open video stream!")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0:
        frame_width, frame_height = 1280, 720

    print(f"Video: {frame_width}x{frame_height}, {fps:.1f} FPS")

    # Setup window
    cv2.namedWindow('Traffic Analysis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Traffic Analysis', 1280, 720)
    cv2.setMouseCallback('Traffic Analysis', mouse_callback)

    tracker = VehicleTracker()

    # Video writer
    output_path = os.path.join(OUTPUT_DIR, f'traffic_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print("\nINSTRUCTIONS:")
    print("  's' - select line points")
    print("  'c' - save data to CSV")
    print("  'q' - quit")
    print("=" * 50)

    frame_count = 0
    last_time = time.time()
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Connection lost, reconnecting...")
            if not USE_WEBCAM:
                stream_url = get_youtube_stream_url(YOUTUBE_STREAM_URL)
                if stream_url:
                    cap = cv2.VideoCapture(stream_url)
            continue

        frame_count += 1

        # Calculate FPS
        if frame_count % 30 == 0:
            current_time = time.time()
            fps_display = 30 / (current_time - last_time)
            last_time = current_time

        # Skip every 2nd frame for performance
        if frame_count % 2 != 0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Draw lines
        cv2.line(frame, LINE1_POINTS[0], LINE1_POINTS[1], (0, 255, 0), 3)
        cv2.line(frame, LINE2_POINTS[0], LINE2_POINTS[1], (0, 255, 0), 3)

        # Line labels
        cv2.putText(frame, "LINE 1", (LINE1_POINTS[0][0], LINE1_POINTS[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "LINE 2", (LINE2_POINTS[0][0], LINE2_POINTS[0][1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # YOLO detection
        results = model.track(frame, conf=CONF_THRESH, tracker=TRACKER,
                              persist=True, verbose=False, classes=[2, 3, 5, 7])

        if results[0].boxes is not None:
            boxes = results[0].boxes

            if boxes.id is not None and boxes.xyxy is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy()
                cls = boxes.cls.cpu().numpy()

                for i, (box, track_id, class_id) in enumerate(zip(xyxy, track_ids, cls)):
                    x1, y1, x2, y2 = box.astype(int)
                    tid = int(track_id)

                    class_name = model.names[int(class_id)]

                    if class_name not in VEHICLE_CLASSES:
                        continue

                    center_y = (y1 + y2) // 2
                    current_time = time.time()

                    # Check line crossing
                    tracker.check_line_crossing(tid, center_y, current_time, class_name)

                    # Calculate speed
                    speed = tracker.calculate_speed(tid, current_time, class_name)

                    # Color by class
                    color = VEHICLE_CLASSES.get(class_name, (0, 255, 0))

                    # Draw box
                    draw_vehicle_box(frame, x1, y1, x2, y2, speed, class_name, tid, color)

                    tracker.total_vehicles.add(tid)

        # Statistics overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y_pos = 40
        cv2.putText(frame, f'TOTAL CROSSED: {len(tracker.vehicles_crossed)}',
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 35
        cv2.putText(frame, f'AVG SPEED: {tracker.avg_speed:.1f} km/h',
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_pos += 35
        cv2.putText(frame, f'UNIQUE VEHICLES: {len(tracker.total_vehicles)}',
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        y_pos += 35
        cv2.putText(frame, f'DISTANCE: {REAL_DISTANCE_METERS} m',
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Point selection mode
        if selecting_points:
            cv2.putText(frame, f'SELECTING LINE {current_line} - CLICK 2 POINTS',
                        (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for pt in points:
                cv2.circle(frame, pt, 6, (0, 0, 255), -1)

        # FPS display
        cv2.putText(frame, f'FPS: {fps_display:.1f}', (frame_width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Connection status
        if not USE_WEBCAM:
            cv2.putText(frame, 'YouTube Live', (frame_width - 200, frame_height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow('Traffic Analysis', frame)
        out.write(frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Shutting down...")
            break
        elif key == ord('s'):
            selecting_points = True
            current_line = 1
            points = []
            print("Point selection mode activated!")
        elif key == ord('c'):
            save_to_csv(tracker)

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "=" * 50)
    print("FINAL STATISTICS")
    print("=" * 50)
    print(f"Total vehicles crossed: {len(tracker.vehicles_crossed)}")
    print(f"Average speed: {tracker.avg_speed:.1f} km/h")
    print(f"Unique vehicles: {len(tracker.total_vehicles)}")

    if tracker.vehicle_speeds:
        print(f"Minimum speed: {min(tracker.vehicle_speeds):.1f} km/h")
        print(f"Maximum speed: {max(tracker.vehicle_speeds):.1f} km/h")

        # Statistics by class
        class_count = {}
        class_speeds = {}
        for data in tracker.crossed_vehicles_data:
            cls = data['class']
            speed = data['speed']
            if cls not in class_count:
                class_count[cls] = 0
                class_speeds[cls] = []
            class_count[cls] += 1
            class_speeds[cls].append(speed)

        print("\nStatistics by vehicle type:")
        for cls in class_count:
            avg_speed_cls = np.mean(class_speeds[cls]) if class_speeds[cls] else 0
            print(f"  {cls}: {class_count[cls]} units, avg speed: {avg_speed_cls:.1f} km/h")

    # Save data
    csv_path = save_to_csv(tracker)
    print(f"Video saved: {output_path}")
    print(f"Data saved: {csv_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()