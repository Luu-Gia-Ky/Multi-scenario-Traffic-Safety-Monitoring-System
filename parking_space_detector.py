# parking_space_detector.py
import os
import cv2
from datetime import datetime
from ultralytics import YOLO

class ParkingSpaceDetector:
    def __init__(self, model_path='yolov8_parkingspace_detectorv2.pt', save_dir='parking_detections'):
        self.model = YOLO(model_path)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def detect(self, frame):
        results = self.model.predict(frame, conf=0.4, iou=0.5)
        detections = []

        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = map(int, box[:6])
                class_id = int(cls)
                if class_id == 1:  # assuming 0 = empty parking space
                    detections.append(((x1, y1, x2, y2), conf))

        return detections

    def annotate(self, frame, detections):
        for (x1, y1, x2, y2), conf in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Empty ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def save_detection(self, frame, detections, timestamp):
        if detections:
            filename = f"parking_{timestamp:.2f}.jpg"
            path = os.path.join(self.save_dir, filename)
            cv2.imwrite(path, frame)
            print(f"📷 Saved parking detection: {path}")
            return path
        return None

