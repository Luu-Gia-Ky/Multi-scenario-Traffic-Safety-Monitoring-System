import os
import cv2

class HelmetDetector:
    def __init__(self, helmet_model, no_helmet_model):
        # Khởi tạo detector với 2 mô hình: đội mũ và không đội mũ
        self.helmet_model = helmet_model  # Model phát hiện người đội mũ
        self.no_helmet_model = no_helmet_model  # Model phát hiện người không đội mũ

        # Các danh sách lưu bounding box
        self.helmet_boxes = []  
        self.no_helmet_boxes = []
        self.saved_boxes = []  # Các box đã lưu ảnh

        self.no_helmet_count = 0  # Số lượng người không đội mũ
        self.helmet_count = 0     # Số lượng người đội mũ

        self.image_folder = 'No_helmet_images'  # Thư mục lưu ảnh người không đội mũ

        self.counting_enabled = True  # Bật/Tắt chế độ đếm
        self.line_ratio = 0.5  # Tỷ lệ vạch ngang (so với chiều cao ảnh)
        self.vertical_line_ratio = 0.5  # Tỷ lệ vạch dọc (so với chiều rộng ảnh)

        self.line_position = None  # Vị trí chính xác của vạch ngang
        self.vertical_line_position = None  # Vị trí chính xác của vạch dọc

        self.already_counted = set()  # Lưu object ID đã đếm

        # Lưu số lượng đếm theo từng lớp
        self.helmet_class_counts = {
            'helmet': 0,
            'no_helmet': 0,
        }

    def reset_counts(self):
        # Reset toàn bộ các bộ đếm về 0
        self.helmet_class_counts = {
            'helmet': 0,
            'no_helmet': 0,
        }
        self.helmet_count = 0
        self.no_helmet_count = 0
        self.already_counted.clear()
        self.helmet_boxes = []
        self.no_helmet_boxes = []
        self.saved_boxes = []

    def enable_counting(self):
        # Bật chế độ đếm đối tượng
        self.counting_enabled = True

    def disable_counting(self):
        # Tắt chế độ đếm đối tượng
        self.counting_enabled = False

    def reset_no_helmet_count(self):
        # Reset số lượng người không đội mũ
        self.no_helmet_count = 0

    def get_no_helmet_count(self):
        # Lấy số lượng người không đội mũ hiện tại
        return self.no_helmet_count

    def get_helmet_count(self):
        # Lấy số lượng người đội mũ hiện tại
        return self.helmet_count

    def extract_box_info(self, box, model):
        # Tách thông tin từ 1 bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tọa độ box
        conf = float(box.conf[0])  # Độ tự tin
        class_id = int(box.cls[0])  # ID lớp
        class_name = model.names[class_id]  # Tên lớp

        # Kiểm tra xem có ID theo dõi không
        object_id = int(box.id[0]) if hasattr(box, 'id') and box.id is not None else None

        return (x1, y1, x2, y2), class_name, conf, object_id

    def detect(self, frame, current_time=None):
        # Phát hiện đối tượng đội và không đội mũ trên frame

        self.helmet_boxes = []
        self.no_helmet_boxes = []
        self.no_helmet_count = 0

        # Phát hiện người đội mũ
        results_helmet = self.helmet_model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        if results_helmet is not None and results_helmet.boxes:
            for box in results_helmet.boxes:
                box_info = self.extract_box_info(box, self.helmet_model)
                if box_info[1].lower() == "helmet":
                    self.helmet_boxes.append(box_info)

                    # Vẽ bounding box và text
                    x1, y1, x2, y2 = box_info[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Helmet {box_info[2]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Phát hiện người không đội mũ
        results_no_helmet = self.no_helmet_model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
        if results_no_helmet is not None and results_no_helmet.boxes:
            for box in results_no_helmet.boxes:
                box_info = self.extract_box_info(box, self.no_helmet_model)
                if box_info[1].lower() == "no_helmet":
                    self.no_helmet_boxes.append(box_info)

                    # Vẽ bounding box và text
                    x1, y1, x2, y2 = box_info[0]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"No Helmet {box_info[2]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Cập nhật số lượng không đội mũ
        self.no_helmet_count = len(self.no_helmet_boxes)

        return frame, self.helmet_boxes, self.no_helmet_boxes

    def save_no_helmet(self, frame, box, current_time):
        # Lưu ảnh người không đội mũ vào thư mục
        if not os.path.exists(self.image_folder):
            os.makedirs(self.image_folder)

        path = os.path.join(self.image_folder, f"{current_time:.2f}_no_helmet.jpg")
        cv2.imwrite(path, frame)  # Lưu ảnh

        self.saved_boxes.append(box)  # Lưu box đã lưu ảnh
        return path

    def count_objects(self, frame, current_time):
        # Đếm đối tượng khi băng qua vạch

        # Tính toán vị trí các vạch
        self.line_position = int(frame.shape[0] * self.line_ratio)
        if self.vertical_line_ratio is not None:
            self.vertical_line_position = int(frame.shape[1] * self.vertical_line_ratio)

        if self.counting_enabled:
            # Vẽ vạch ngang
            cv2.line(frame, (0, self.line_position), (frame.shape[1], self.line_position), (0, 0, 255), 2)

            # Vẽ vạch dọc
            if self.vertical_line_position is not None:
                cv2.line(frame, (self.vertical_line_position, 0), (self.vertical_line_position, frame.shape[0]), (255, 0, 0), 2)

            # Hiển thị số lượng đã đếm
            cv2.putText(frame, f'No Helmet : {self.helmet_class_counts["no_helmet"]}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f'Helmet : {self.helmet_class_counts["helmet"]}', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Duyệt từng box đã detect
        for (box, class_name, _, object_id) in self.helmet_boxes + self.no_helmet_boxes:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Vẽ tâm đối tượng
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            if object_id is None or object_id in self.already_counted:
                continue  # Bỏ qua nếu đã đếm

            crossed_line = False

            # Kiểm tra băng qua vạch ngang
            if self.line_position - 10 < center_y < self.line_position + 10:
                crossed_line = True

            # Kiểm tra băng qua vạch dọc
            if self.vertical_line_position is not None and self.vertical_line_position - 10 < center_x < self.vertical_line_position + 10:
                crossed_line = True

            if crossed_line:
                self.already_counted.add(object_id)  # Đánh dấu đã đếm
                if class_name.lower() == "helmet":
                    self.helmet_class_counts["helmet"] += 1
                elif class_name.lower() == "no_helmet":
                    self.helmet_class_counts["no_helmet"] += 1
                    self.save_no_helmet(frame, box, current_time)  # Lưu ảnh người không đội mũ
