# 🚦 Multi-scenario Traffic Safety Monitoring System
## Hệ Thống Giám Sát An Toàn Giao Thông Đa Kịch Bản

![Project Banner](./traffic_safety_monitoring_banner_1778574315741.png)

## 🌟 Overview / Tổng quan
This project is an advanced traffic monitoring system powered by **YOLOv8**, designed to enhance road safety through real-time computer vision. It supports multiple scenarios including vehicle tracking, accident detection, helmet compliance, and parking management.

Dự án này là một hệ thống giám sát giao thông tiên tiến sử dụng **YOLOv8**, được thiết kế để tăng cường an toàn đường bộ thông qua thị giác máy tính thời gian thực. Hệ thống hỗ trợ nhiều kịch bản bao gồm theo dõi phương tiện, phát hiện tai nạn, kiểm tra mũ bảo hiểm và quản lý bãi đỗ xe.

---

## ✨ Key Features / Tính năng chính

### 🏎️ Vehicle Tracking & Counting (Theo dõi & Đếm phương tiện)
- Real-time detection and tracking of various vehicle classes.
- Automated counting and speed estimation using virtual lines.
- *File: `vehicle_tracker.py`*

### ⚠️ Accident Detection (Phát hiện tai nạn)
- Instant identification of road accidents using specialized YOLO models.
- Visual alerts and location flagging in the video stream.
- *File: `accident_detector.py`*

### ⛑️ Helmet Compliance (Kiểm tra mũ bảo hiểm)
- Detects riders with and without helmets.
- Ideal for urban safety enforcement and automated ticketing systems.
- *File: `helmet.py`*

### 🅿️ Parking Space Detection (Phát hiện bãi đỗ xe trống)
- Monitors parking lots to identify available spaces in real-time.
- Visual overlays showing occupied vs. free spots.
- *File: `parking_space_detector.py`*

### 📊 Automated Reporting (Báo cáo tự động)
- Generates detailed PDF reports summarizing detections and safety violations.
- *File: `report_generator.py`*

---

## 🛠️ Tech Stack / Công nghệ sử dụng
- **Language:** Python 3.x
- **Computer Vision:** OpenCV, Ultralytics YOLOv8
- **GUI/Web:** Streamlit (Web interface), Tkinter (Legacy Desktop GUI)
- **Data:** NumPy, PIL
- **Deployment:** Docker, Docker Compose
- **Reporting:** FPDF

---

## 🚀 Getting Started / Bắt đầu

### Prerequisites / Yêu cầu hệ thống
- Python 3.8+
- Webcam or Video files for testing

### Installation / Cài đặt
1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Multi-scenario-Traffic-Safety-Monitoring-System
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Models:**
   Ensure the following `.pt` files are in the root directory:
   - `yolov8n.pt`
   - `yolov8_accident_model.pt`
   - `yolov8_helmet_model2.pt`
   - `yolov8_no_helmet_model.pt`
   - `yolov8_parkingspace_detectorv2.pt`

### Running the App / Khởi chạy

#### 🌐 Web Interface (Recommended)
Run the Streamlit application for a modern web-based experience:
```bash
streamlit run app_web.py
```

#### 🖥️ Desktop GUI
Run the legacy Tkinter interface:
```bash
python giaodien.py
```

---

## 🐳 Docker Support / Hỗ trợ Docker
Deploy the system easily using Docker:

```bash
docker-compose up --build
```

---

## 📁 Project Structure / Cấu trúc thư mục
- `app_web.py`: Main entry point for the Streamlit web app.
- `giaodien.py`: Main entry point for the Tkinter desktop app.
- `accident_detector.py`: Accident detection logic.
- `helmet.py`: Helmet compliance monitoring.
- `vehicle_tracker.py`: Vehicle tracking and counting.
- `parking_space_detector.py`: Parking availability analysis.
- `report_generator.py`: PDF report generation module.
- `accident_images/`: Storage for captured accident frames.
- `No_helmet_images/`: Storage for helmet violation frames.
- `pdf_report/`: Directory for generated safety reports.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Developed with ❤️ for Traffic Safety.*
