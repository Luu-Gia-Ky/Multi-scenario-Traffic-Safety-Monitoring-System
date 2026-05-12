from fpdf import FPDF

class ReportGenerator:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=12)

    def add_accident(self, time, class_name, box, image_path):
        x1, y1, x2, y2 = box
        self.pdf.add_page()
        self.pdf.cell(200, 10, txt=f"Accident at {time:.2f} seconds", ln=True, align='C')
        self.pdf.ln(5)
        self.pdf.cell(200, 10, txt=f"Type: {class_name}", ln=True)
        self.pdf.cell(200, 10, txt=f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})", ln=True)
        self.pdf.image(image_path, x=10, w=180)

    def add_red_light_violation(self, time, vehicle_id, box, image_path):
        x1, y1, x2, y2 = box
        self.pdf.add_page()
        self.pdf.cell(200, 10, txt=f"Red Light Violation at {time:.2f} seconds", ln=True, align='C')
        self.pdf.ln(5)
        self.pdf.cell(200, 10, txt=f"Vehicle ID: {vehicle_id}", ln=True)
        self.pdf.cell(200, 10, txt=f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})", ln=True)
        self.pdf.image(image_path, x=10, w=180)

    def add_no_helmet(self, time, class_name, box, image_path):
        x1, y1, x2, y2 = box
        self.pdf.add_page()
        self.pdf.cell(200, 10, txt=f"No_helmet at {time:.2f} seconds", ln=True, align='C')
        self.pdf.ln(5)
        self.pdf.cell(200, 10, txt=f"Type: {class_name}", ln=True)
        self.pdf.cell(200, 10, txt=f"Bounding Box: ({x1}, {y1}) to ({x2}, {y2})", ln=True)
        self.pdf.image(image_path, x=10, w=180)
    def save(self, path):
        self.pdf.output(path)
