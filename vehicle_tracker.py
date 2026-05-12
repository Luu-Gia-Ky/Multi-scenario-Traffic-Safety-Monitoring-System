import numpy as np
import cv2

class VehicleTracker:
    def __init__(self, model):
        self.model = model
        self.previous_positions = {}
        self.speed_history = {}
        self.noise_threshold = 0.15
        self.counting_enabled = True

        self.line_ratio = 0.5
        self.line_position = None

        self.vertical_line_ratio = 0.5
        self.vertical_line_position = None

        self.vehicle_count = 0
        self.already_counted = set()

        self.vehicle_class_counts = {
            'car': 0,
            'truck': 0,
            'bus': 0,
            'motorcycle': 0
        }

        self.speed_limit_threshold = 30  # 20 km/h

    def enable_counting(self):
        self.counting_enabled = True

    def disable_counting(self):
        self.counting_enabled = False

    def track(self, frame, current_time):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml")
        detections = []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = self.model.names[class_id]
                object_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else f"{class_name}_{x1}_{y1}"
                detections.append(((x1, y1, x2, y2), class_name, confidence, object_id))

        return detections

    def annotate_speed(self, frame, current_time, detections, processed_ids):
        # Tính vị trí dòng ngang để đếm phương tiện (theo tỷ lệ chiều cao frame)
        self.line_position = int(frame.shape[0] * self.line_ratio)

        if self.vertical_line_ratio is not None:
            # Nếu có yêu cầu, tính vị trí dòng dọc để đếm
            self.vertical_line_position = int(frame.shape[1] * self.vertical_line_ratio)

        if self.counting_enabled:
            # Vẽ dòng ngang màu đỏ
            cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)

            if self.vertical_line_position is not None:
                # Vẽ dòng dọc màu xanh nếu có
                cv2.line(frame, (self.vertical_line_position, 0), (self.vertical_line_position, frame.shape[0]), (255, 0, 0), 2)

            # Hiển thị tổng số phương tiện đếm được
            cv2.putText(frame, f'Vehicle sum: {self.vehicle_count}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)

            # Hiển thị số lượng từng loại xe (car, truck, bus, motorcycle)
            y_offset = 80
            for cls, count in self.vehicle_class_counts.items():
                cv2.putText(frame, f'{cls.capitalize()}: {count}', (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 225), 2)
                y_offset += 30

        # Xử lý từng detection
        for (x1, y1, x2, y2), class_name, _, object_id in detections:
            # Bỏ qua những object đã xử lý hoặc không phải phương tiện quan tâm
            if object_id in processed_ids or class_name not in ['car', 'truck', 'bus', 'motorcycle']:
                continue

            # Thiết lập tỷ lệ đổi pixel -> mét (xe máy nhỏ hơn ô tô)
            pixel_to_meter = 0.05 if class_name == 'motorcycle' else 0.1

            # Tính tâm của bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if object_id in self.previous_positions:
                # Nếu đã lưu vị trí cũ của object
                px, py, pt, _ = self.previous_positions[object_id]

                # Tính quãng đường di chuyển (pixel -> meter)
                distance = np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2) * pixel_to_meter

                # Tính thời gian đã đi
                time_diff = current_time - pt

                if time_diff > 0:
                    # Tính tốc độ: quãng đường / thời gian, đổi m/s -> km/h
                    speed = (distance / time_diff) * 3.6

                    if distance < self.noise_threshold:
                        # Nếu quãng đường quá nhỏ => bỏ, tránh noise
                        speed = 0

                    if speed > self.speed_limit_threshold:
                        # Nếu tốc độ tính quá cao => giảm pixel_to_meter lại (để giảm tốc độ tính)
                        pixel_to_meter = 0.02 if class_name == 'motorcycle' else 0.04
                        distance = np.sqrt((center_x - px) ** 2 + (center_y - py) ** 2) * pixel_to_meter
                        speed = (distance / time_diff) * 3.6 * 0.7  # Giảm thêm 30%
                    elif speed < self.speed_limit_threshold:
                        # Nếu tốc độ quá thấp => tăng 50% tốc độ
                        speed = speed * 2

                    # Lưu lịch sử tốc độ để làm mượt giá trị trung bình
                    self.speed_history.setdefault(object_id, []).append(speed)
                    if len(self.speed_history[object_id]) > 5:
                        self.speed_history[object_id].pop(0)

                    # Tính trung bình trọng số, giá trị mới nhất nặng hơn
                    avg_speed = np.average(
                        self.speed_history[object_id],
                        weights=np.linspace(1, 0.5, len(self.speed_history[object_id]))
                    )

                    if avg_speed < 0.5:
                        # Nếu trung bình quá nhỏ, coi như bằng 0
                        avg_speed = 0

                    # Đặt màu theo loại phương tiện
                    color = (255, 0, 0) if class_name == 'motorcycle' else (0, 255, 0)

                    # Vẽ tốc độ lên khung hình
                    cv2.putText(frame, f'{int(avg_speed)} km/h', (x1, y1 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Kiểm tra xem phương tiện đã băng qua vạch chưa để đếm
            if object_id in self.previous_positions:
                px, py, pt, _ = self.previous_positions[object_id]
                crossed_line = False

                # Kiểm tra vượt dòng ngang
                if py < self.line_position <= center_y or py > self.line_position >= center_y:
                    crossed_line = True

                # Kiểm tra vượt dòng dọc nếu có
                if self.vertical_line_position is not None:
                    if px < self.vertical_line_position <= center_x or px > self.vertical_line_position >= center_x:
                        crossed_line = True

                if crossed_line and object_id not in self.already_counted:
                    # Nếu vượt vạch lần đầu tiên => đếm
                    self.vehicle_count += 1
                    self.already_counted.add(object_id)

                    # Tăng đếm từng loại xe
                    if class_name in self.vehicle_class_counts:
                        self.vehicle_class_counts[class_name] += 1

                    # In thông tin ra console
                    print(f'Xe đã đếm: {class_name} | ID: {object_id} | Tổng: {self.vehicle_count}')

            # Cập nhật vị trí mới của object
            self.previous_positions[object_id] = (center_x, center_y, current_time, class_name)

            # Vẽ bounding box và tên phương tiện
            color = (255, 0, 0) if class_name == 'motorcycle' else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
