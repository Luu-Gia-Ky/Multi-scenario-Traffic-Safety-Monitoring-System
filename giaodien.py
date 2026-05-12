#  Import các module cần thiết 
import tkinter as tk  # Thư viện Tkinter để tạo giao diện
from tkinter import filedialog, messagebox  # Để mở hộp thoại chọn file và hiển thị thông báo
import cv2  # Thư viện OpenCV để xử lý video
from PIL import Image, ImageTk  # Để chuyển đổi ảnh OpenCV sang Tkinter
import os  # Thao tác với hệ thống file
import time  # Lấy thời gian hiện tại

# Import các mô-đun xử lý riêng
from ultralytics import YOLO  # Mô hình YOLO
from accident_detector import AccidentDetector  # Phát hiện tai nạn
from helmet import HelmetDetector  # Phát hiện mũ bảo hiểm
from vehicle_tracker import VehicleTracker  # Theo dõi phương tiện
from report_generator import ReportGenerator  # Tạo báo cáo (chưa sử dụng trong đoạn này)
from parking_space_detector import ParkingSpaceDetector  # Phát hiện chỗ đỗ xe (chưa sử dụng trong đoạn này)

#  Load mô hình YOLO 
vehicle_model = YOLO('yolov8n.pt')  # Mô hình phát hiện phương tiện
accident_model = YOLO('yolov8_accident_model.pt')  # Mô hình phát hiện tai nạn
helmet_model = YOLO('yolov8_helmet_model2.pt')  # Mô hình phát hiện người đội mũ bảo hiểm
no_helmet_model = YOLO('yolov8_no_helmet_model.pt')  # Mô hình phát hiện người không đội mũ bảo hiểm
parking_model_path = 'yolov8_parkingspace_detectorv2.pt'  # Mô hình phát hiện chỗ đỗ xe

#  Khởi tạo các đối tượng xử lý 
vehicle_tracker = VehicleTracker(vehicle_model)
helmet_detector = HelmetDetector(helmet_model, no_helmet_model)
accident_detector = AccidentDetector(accident_model)
report_generator = ReportGenerator()
parking_detector = ParkingSpaceDetector()

#  Giao diện người dùng 
left_buttons = []  # Lưu các nút bên trái
left_buttons2 = []  # Lưu các nút thay đổi vạch ngang
left_buttons3 = []  # Lưu các nút thay đổi vạch dọc

root = tk.Tk()  # Khởi tạo cửa sổ
root.title("Hệ thống giám sát giao thông")  # Đặt tiêu đề
root.geometry("1400x800")  # Kích thước cửa sổ

selected_mode = tk.IntVar(value=1)  # Lưu chế độ hiện tại (mặc định 1)

cap = None  # Đối tượng VideoCapture
video_path = ""  # Đường dẫn video gốc
out = None  # Đối tượng VideoWriter
save_video_path = ""  # Đường dẫn lưu video

# Biến kéo vạch
is_dragging_horizontal = False
is_dragging_vertical = False

#  Thiết kế layout giao diện 
left_frame = tk.Frame(root, width=150, height=800, bg="white", relief="solid", bd=1)
left_frame.pack(side="left", fill="y")

video_frame = tk.Label(root, bg="lightgray")
video_frame.pack(side="left", expand=True, fill="both")

info_frame = tk.Frame(root, width=250, bg="white", relief="solid", bd=1)
info_frame.pack(side="right", fill="y")

# Bind các sự kiện kéo thả chuột để chỉnh vạch
video_frame.bind("<Button-1>", lambda e: start_drag(e))
video_frame.bind("<B1-Motion>", lambda e: drag_line(e))
video_frame.bind("<ButtonRelease-1>", lambda e: stop_drag(e))

# Thông tin số lượng phương tiện và mũ bảo hiểm bên phải
tk.Label(info_frame, text="Số phương tiện xuất hiện", bg="white", anchor="w").pack(anchor="w", padx=10, pady=5)
total_vehicle_label = tk.Label(info_frame, text="Tổng phương tiện: 0", bg="white", anchor="w")
total_vehicle_label.pack(anchor="w", padx=10, pady=5)

vehicle_labels = {}
for cls in ['car', 'truck', 'bus', 'motorcycle']:
    lbl = tk.Label(info_frame, text=f"{cls}: 0", bg="white", anchor="w")
    lbl.pack(anchor="w", padx=10)
    vehicle_labels[cls] = lbl

tk.Label(info_frame, text="Kiểm tra mũ bảo hiểm ", bg="white", anchor="w").pack(anchor="w", padx=10, pady=5)
no_helmet_label = tk.Label(info_frame, text="No helmet: 0", bg="white", anchor="w")
no_helmet_label.pack(anchor="w", padx=10, pady=10)
helmet_label = tk.Label(info_frame, text="Helmet: 0", bg="white", anchor="w")
helmet_label.pack(anchor="w", padx=10, pady=10)

#  Các hàm chức năng 

# Cho phép kéo vạch ngang
def enable_dragging_horizontal():
    global is_dragging_horizontal
    is_dragging_horizontal = True

# Cho phép kéo vạch dọc
def enable_dragging_vertical():
    global is_dragging_vertical
    is_dragging_vertical = True

# Ngừng kéo vạch ngang
def disable_dragging_horizontal():
    global is_dragging_horizontal
    is_dragging_horizontal = False

# Ngừng kéo vạch dọc
def disable_dragging_vertical():
    global is_dragging_vertical
    is_dragging_vertical = False

# Bắt đầu kéo
def start_drag(event):
    if is_dragging_horizontal or is_dragging_vertical:
        update_line_position(event)

# Kéo vạch
def drag_line(event):
    if is_dragging_horizontal or is_dragging_vertical:
        update_line_position(event)

# Dừng kéo vạch
def stop_drag(event):
    pass

# Cập nhật vị trí vạch khi kéo
def update_line_position(event):
    frame_height = 600
    frame_width = 800

    if is_dragging_horizontal:
        y = max(0, min(event.y, frame_height))
        vehicle_tracker.line_ratio = y / frame_height
        vehicle_tracker.line_position = int(vehicle_tracker.line_ratio * frame_height)
        helmet_detector.line_ratio = y / frame_height
        helmet_detector.line_position = int(helmet_detector.line_ratio * frame_height)

    if is_dragging_vertical:
        x = max(0, min(event.x, frame_width))
        vehicle_tracker.vertical_line_ratio = x / frame_width
        vehicle_tracker.vertical_line_position = int(vehicle_tracker.vertical_line_ratio * frame_width)
        helmet_detector.vertical_line_ratio = x / frame_width
        helmet_detector.vertical_line_position = int(helmet_detector.vertical_line_ratio * frame_width)

# Reset bộ đếm phương tiện và mũ bảo hiểm
def reset_counter():
    for cls in ['car', 'bus', 'truck', 'motorcycle']:
        vehicle_tracker.vehicle_class_counts[cls] = 0
        vehicle_labels[cls].config(text=f"{cls}: 0")
    vehicle_tracker.vehicle_count = 0
    total_vehicle_label.config(text="Tổng phương tiện: 0")
    helmet_detector.reset_counts()
    no_helmet_label.config(text="No helmet: 0")
    helmet_label.config(text="Helmet: 0")

# Highlight nút khi bấm
def highlight_button(clicked_button):
    for btn in left_buttons:
        btn.configure(bg="SystemButtonFace")
    clicked_button.configure(bg="lightblue")

def highlight_button2(clicked_button):
    for btn in left_buttons2:
        btn.configure(bg="SystemButtonFace")
    clicked_button.configure(bg="red")

def highlight_button3(clicked_button):
    for btn in left_buttons3:
        btn.configure(bg="SystemButtonFace")
    clicked_button.configure(bg="blue")

# Mở video từ file
def choose_video():
    global video_path, cap
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        reset_counter()
        messagebox.showinfo("Video đã chọn", video_path)

# Dừng video và lưu kết quả
def stop_video():
    global cap, out
    if cap:
        cap.release()
    if out:
        out.release()
        messagebox.showinfo("Video đã lưu", f"Video đã được lưu tại: {save_video_path}")
    reset_counter()

# Dừng video không thông báo
def stop_video2():
    global cap, out
    if cap:
        cap.release()
    if out:
        out.release()
    reset_counter()

# Cập nhật chế độ xử lý
def update_mode(val):
    selected_mode.set(val)
    print("Chế độ được chọn:", val)

# Chạy video
def run_video():
    global cap, out, save_video_path
    if not cap or not cap.isOpened():
        messagebox.showerror("Lỗi", "Chưa chọn video hoặc video không hợp lệ")
        return

    save_folder = "outputvideo/"
    os.makedirs(save_folder, exist_ok=True)
    filename = time.strftime("%Y%m%d-%H%M%S") + ".avi"
    save_video_path = os.path.join(save_folder, filename)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (1200, 800))

    read_frame()

# Cập nhật số lượng phương tiện
def update_vehicle_counts():
    counts = vehicle_tracker.vehicle_class_counts
    for cls in ['car', 'bus', 'truck', 'motorcycle']:
        vehicle_labels[cls].config(text=f"{cls}: {counts.get(cls, 0)}")
    total_vehicle_label.config(text=f"Tổng phương tiện: {vehicle_tracker.vehicle_count}")

# Gộp nhiều box thành 1
def merge_boxes(boxes):
    x1 = min([box[0] for box in boxes])
    y1 = min([box[1] for box in boxes])
    x2 = max([box[2] for box in boxes])
    y2 = max([box[3] for box in boxes])
    return (x1, y1, x2, y2)

# Đọc từng frame
def read_frame():
    global cap, out
    ret, frame = cap.read()
    if not ret:
        if cap:
            cap.release()
        if out:
            out.release()
            messagebox.showinfo("Hoàn tất", f"Video đã được lưu tại: {save_video_path}")
        return

    frame = cv2.resize(frame, (1200, 800))

    if out is None:
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_video_path, fourcc, 20.0, (width, height))

    current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    mode = selected_mode.get()

    if mode == 1:
        detections = vehicle_tracker.track(frame, current_time)
        vehicle_tracker.annotate_speed(frame, current_time, detections, set())
        update_vehicle_counts()

    elif mode == 2:
        frame, helmet_boxes, no_helmet_boxes = helmet_detector.detect(frame, current_time)
        helmet_detector.count_objects(frame, current_time)
        no_helmet_label.config(text=f"No helmet: {helmet_detector.helmet_class_counts['no_helmet']}")
        helmet_label.config(text=f"Helmet: {helmet_detector.helmet_class_counts['helmet']}")

    elif mode == 3:
        vehicle_tracker.disable_counting()
        detections = vehicle_tracker.track(frame, current_time)
        accident_boxes = [
            (box, cls_name, conf)
            for (box, cls_name, conf) in accident_detector.detect(frame)
            if conf > 0.7
        ]
        if accident_boxes:
            update_vehicle_counts()
        else:
            vehicle_tracker.annotate_speed(frame, current_time, detections, set())
            update_vehicle_counts()

        processed_ids = set()
        for (x1, y1, x2, y2), cls_name, conf in accident_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f'{cls_name.upper()} ({conf:.2f})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif mode == 4:
        parking_detections = parking_detector.detect(frame)
        parking_detector.annotate(frame, parking_detections)
        parking_detector.save_detection(frame, parking_detections, current_time)
    if out:
        out.write(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_frame.imgtk = img
    video_frame.configure(image=img)

    root.after(30, read_frame)

#  Các nút giao diện 
btn_video = tk.Button(left_frame, text="Video cần kiểm tra", command=lambda: [choose_video()])
btn_video.pack(pady=5)
left_buttons.append(btn_video)

tk.Label(left_frame, text="Chọn chế độ xử lý:", font=("Arial", 12, "bold")).pack(pady=5)

btn_count = tk.Button(left_frame, text="Kiểm tra số lượng xe", command=lambda: [update_mode(1), highlight_button(btn_count)])
btn_count.pack(pady=2)
left_buttons.append(btn_count)

btn_helmet = tk.Button(left_frame, text="Kiểm tra đội mũ bảo hiểm", command=lambda: [update_mode(2), highlight_button(btn_helmet)])
btn_helmet.pack(pady=2)
left_buttons.append(btn_helmet)

btn_accident = tk.Button(left_frame, text="Phát hiện Tai nạn", command=lambda: [update_mode(3), highlight_button(btn_accident)])
btn_accident.pack(pady=2)
left_buttons.append(btn_accident)

btn_parking = tk.Button(left_frame, text="Phát hiện bãi đỗ xe trống", command=lambda: [update_mode(4), highlight_button(btn_parking)])
btn_parking.pack(pady=2)
left_buttons.append(btn_parking)


tk.Label(left_frame, text="Thay đổi vạch ngang", font=("Arial", 12, "bold")).pack(pady=5)

line_buttons_frame = tk.Frame(left_frame)
line_buttons_frame.pack(pady=2)

btn_increase_line = tk.Button(line_buttons_frame, text="Bật", command=lambda: [enable_dragging_horizontal(), highlight_button2(btn_increase_line)])
btn_increase_line.pack(side="left", padx=5)
left_buttons2.append(btn_increase_line)

btn_decrease_line = tk.Button(line_buttons_frame, text="Tắt", command=lambda: [disable_dragging_horizontal(), highlight_button2(btn_decrease_line)])
btn_decrease_line.pack(side="left", padx=5)
left_buttons2.append(btn_decrease_line)

tk.Label(left_frame, text="Thay đổi vạch dọc", font=("Arial", 12, "bold")).pack(pady=5)

line_buttons_frame = tk.Frame(left_frame)
line_buttons_frame.pack(pady=2)

btn_increase_line2 = tk.Button(line_buttons_frame, text="Bật", command=lambda: [enable_dragging_vertical(), highlight_button3(btn_increase_line2)])
btn_increase_line2.pack(side="left", padx=5)
left_buttons3.append(btn_increase_line2)

btn_decrease_line2 = tk.Button(line_buttons_frame, text="Tắt", command=lambda: [disable_dragging_vertical(), highlight_button3(btn_decrease_line2)])
btn_decrease_line2.pack(side="left", padx=5)
left_buttons3.append(btn_decrease_line2)

control_frame = tk.Frame(left_frame, bg="white")
control_frame.pack(pady=10)

btn_start = tk.Button(control_frame, text="Bắt đầu", command=run_video)
btn_start.pack(pady=2)

btn_stop = tk.Button(control_frame, text="Dừng", command=stop_video2)
btn_stop.pack(pady=2)

btn_exit = tk.Button(left_frame, text="Thoát", command=lambda: [root.quit(), highlight_button(btn_exit)])
btn_exit.pack(pady=5)
left_buttons.append(btn_exit)

#  Chạy giao diện 
root.mainloop()
