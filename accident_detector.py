import os
import cv2

class AccidentDetector:
    def __init__(self, model):
        self.model = model  # Gán mô hình phát hiện tai nạn (ví dụ YOLO) cho đối tượng
        self.saved_accidents = []  # Khởi tạo danh sách các bounding box tai nạn đã lưu để kiểm tra trùng lặp
        self.image_folder = 'accident_images'  # Tên thư mục lưu ảnh các vụ tai nạn
        os.makedirs(self.image_folder, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

    def detect(self, frame, conf_threshold=0.5):
        """
        Phát hiện các đối tượng tai nạn trong frame đầu vào.

        Args:
            frame: Ảnh đầu vào (numpy array).
            conf_threshold: Ngưỡng confidence để lọc bớt kết quả kém chính xác.

        Returns:
            boxes: Danh sách các bounding box dạng [(x1, y1, x2, y2), class_name, confidence]
        """
        results = self.model.predict(frame, conf=conf_threshold, verbose=False)  # Dự đoán kết quả từ mô hình

        boxes = []  # Danh sách lưu các box hợp lệ

        for box in results[0].boxes:  # Duyệt qua tất cả box mà mô hình trả về
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ 4 góc box, ép kiểu int
            conf = float(box.conf[0])  # Lấy độ tự tin (confidence) của box

            if conf < conf_threshold:
                continue  # Bỏ qua nếu độ tự tin thấp hơn ngưỡng

            class_id = int(box.cls[0])  # ID của lớp dự đoán
            class_name = self.model.names[class_id]  # Tên lớp dự đoán (từ dict tên lớp)

            boxes.append(((x1, y1, x2, y2), class_name, conf))  # Thêm box vào danh sách kết quả

        return boxes  # Trả về danh sách các box

    def is_duplicate(self, new_box, threshold=0.3):
        """
        Kiểm tra một bounding box mới có trùng với các box đã lưu trước đó không,
        dựa vào chỉ số IoU (Intersection over Union).

        Args:
            new_box: Bounding box mới (x1, y1, x2, y2).
            threshold: Ngưỡng IoU để xác định trùng lặp.

        Returns:
            True nếu trùng, False nếu không.
        """
        x1, y1, x2, y2 = new_box
        new_area = (x2 - x1) * (y2 - y1)  # Tính diện tích box mới

        for (sx1, sy1, sx2, sy2) in self.saved_accidents:  # Duyệt qua từng box đã lưu
            # Tính tọa độ giao nhau giữa hai box
            inter_x1 = max(x1, sx1)
            inter_y1 = max(y1, sy1)
            inter_x2 = min(x2, sx2)
            inter_y2 = min(y2, sy2)

            # Tính diện tích phần giao nhau (nếu có)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

            # Tính diện tích hợp nhất (union)
            union_area = new_area + (sx2 - sx1) * (sy2 - sy1) - inter_area

            # Tính chỉ số IoU
            iou = inter_area / (union_area + 1e-5)  # +1e-5 để tránh chia cho 0

            if iou > threshold:
                return True  # Trả về True nếu IoU vượt ngưỡng => trùng lặp

        return False  # Nếu kiểm tra hết không trùng, trả về False

    def save_accident(self, frame, box, current_time):
        """ 
        Lưu ảnh hiện tại chứa vụ tai nạn vào file và cập nhật danh sách box đã lưu.

        Args:
            frame: Ảnh hiện tại (numpy array).
            box: Bounding box tương ứng (x1, y1, x2, y2).
            current_time: Thời gian hiện tại (dùng để đặt tên file ảnh).

        Returns:
            path: Đường dẫn tới file ảnh vừa lưu.
        """
        path = f"{self.image_folder}/{current_time:.2f}_accident.jpg"  # Tạo đường dẫn lưu ảnh, lấy 2 chữ số thập phân
        cv2.imwrite(path, frame)  # Lưu frame vào file ảnh
        self.saved_accidents.append(box)  # Thêm box vào danh sách đã lưu để kiểm tra trùng lặp sau này
        return path  # Trả về đường dẫn file ảnh
