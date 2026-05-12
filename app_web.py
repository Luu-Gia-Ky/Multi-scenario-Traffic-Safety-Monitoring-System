import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import time
import tempfile

# Import các mô-đun xử lý của bạn
from ultralytics import YOLO
from accident_detector import AccidentDetector
from helmet import HelmetDetector
from vehicle_tracker import VehicleTracker
from report_generator import ReportGenerator
from parking_space_detector import ParkingSpaceDetector

# Cấu hình trang Streamlit
st.set_page_config(page_title="Hệ thống giám sát giao thông", layout="wide")

# Hàm load model (dùng cache để không load lại nhiều lần)
@st.cache_resource
def load_models():
    models = {
        "vehicle": YOLO('yolov8n.pt'),
        "accident": YOLO('yolov8_accident_model.pt'),
        "helmet": YOLO('yolov8_helmet_model2.pt'),
        "no_helmet": YOLO('yolov8_no_helmet_model.pt'),
        "parking": 'yolov8_parkingspace_detectorv2.pt'
    }
    return models

models = load_models()

# Khởi tạo các đối tượng xử lý
vehicle_tracker = VehicleTracker(models["vehicle"])
helmet_detector = HelmetDetector(models["helmet"], models["no_helmet"])
accident_detector = AccidentDetector(models["accident"])
parking_detector = ParkingSpaceDetector()

# --- GIAO DIỆN SIDEBAR ---
st.sidebar.title("Cấu hình hệ thống")
mode = st.sidebar.selectbox(
    "Chọn chế độ xử lý:",
    ("Kiểm tra số lượng xe", "Kiểm tra đội mũ bảo hiểm", "Phát hiện Tai nạn", "Phát hiện bãi đỗ xe trống")
)

st.sidebar.markdown("### Điều chỉnh vạch đếm")
line_h = st.sidebar.slider("Vị trí vạch ngang (%)", 0, 100, 50)
line_v = st.sidebar.slider("Vị trí vạch dọc (%)", 0, 100, 50)

uploaded_file = st.sidebar.file_uploader("Tải lên video kiểm tra", type=["mp4", "avi", "mov"])

st.sidebar.markdown("---")
st.sidebar.info("Hệ thống giám sát giao thông thông minh sử dụng YOLOv8.")

# --- GIAO DIỆN CHÍNH ---
st.title("🚦 " + mode)

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Luồng Video")
    video_placeholder = st.empty()

with col2:
    st.subheader("Thống kê")
    stats_placeholder = st.empty()

if uploaded_file is not None:
    # Lưu video tạm thời để OpenCV có thể đọc
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Nút điều khiển
    run_button = st.sidebar.button("Bắt đầu xử lý")
    stop_button = st.sidebar.button("Dừng lại")

    if run_button:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1200, 800))
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Cập nhật vị trí vạch từ thanh trượt
            vehicle_tracker.line_ratio = line_h / 100.0
            vehicle_tracker.vertical_line_ratio = line_v / 100.0
            helmet_detector.line_ratio = line_h / 100.0
            helmet_detector.vertical_line_ratio = line_v / 100.0
            
            # Xử lý theo chế độ được chọn
            if mode == "Kiểm tra số lượng xe":
                detections = vehicle_tracker.track(frame, current_time)
                vehicle_tracker.annotate_speed(frame, current_time, detections, set())
                
                counts = vehicle_tracker.vehicle_class_counts
                stats_text = f"**Tổng phương tiện:** {vehicle_tracker.vehicle_count}\n\n"
                for cls, count in counts.items():
                    stats_text += f"- {cls.capitalize()}: {count}\n"
                stats_placeholder.markdown(stats_text)

            elif mode == "Kiểm tra đội mũ bảo hiểm":
                frame, _, _ = helmet_detector.detect(frame, current_time)
                helmet_detector.count_objects(frame, current_time)
                
                stats_text = f"**Thống kê mũ bảo hiểm:**\n\n"
                stats_text += f"- No Helmet: {helmet_detector.helmet_class_counts['no_helmet']}\n"
                stats_text += f"- Helmet: {helmet_detector.helmet_class_counts['helmet']}\n"
                stats_placeholder.markdown(stats_text)

            elif mode == "Phát hiện Tai nạn":
                detections = vehicle_tracker.track(frame, current_time)
                accident_boxes = accident_detector.detect(frame)
                
                if accident_boxes:
                    for (x1, y1, x2, y2), cls_name, conf in accident_boxes:
                        if conf > 0.7:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "TAI NAN!", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                else:
                    vehicle_tracker.annotate_speed(frame, current_time, detections, set())
                
                stats_placeholder.warning("Đang giám sát tai nạn...")

            elif mode == "Phát hiện bãi đỗ xe trống":
                parking_detections = parking_detector.detect(frame)
                parking_detector.annotate(frame, parking_detections)
                stats_placeholder.success(f"Chỗ trống tìm thấy: {len(parking_detections)}")

            # Hiển thị frame lên Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            if stop_button:
                break

        cap.release()
        st.sidebar.success("Đã hoàn tất xử lý video.")
else:
    st.info("Vui lòng tải lên một file video từ thanh công cụ bên trái để bắt đầu.")
