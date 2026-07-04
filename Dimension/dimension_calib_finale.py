"""
Dimension Calibration & Measurement App
========================================

Two calibration methods:
  Method 1 - Ruler/Scale: draw a line across two known marks on a ruler
             (e.g. 0 mm to 10 mm), enter the real distance, get mm/px.
  Method 2 - ArUco Marker: enter the marker's printed length & breadth (mm),
             auto-detect the marker corners, get mm/px.

After calibration, click "Detect & Measure" to find the largest object's
Outer Diameter (OD) and Inner Diameter (ID) using Otsu + contour ellipse fit.
ArUco tag region is excluded from object detection.

Requires:  PyQt5, opencv-contrib-python, numpy
    pip install PyQt5 opencv-contrib-python numpy
"""

import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton, QDoubleSpinBox,
    QComboBox, QStackedWidget, QMessageBox, QPlainTextEdit,
    QSlider, QCheckBox,
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal


# ============================================================
# ArUco compatibility wrapper (handles OpenCV 4.6 vs 4.7+ API)
# ============================================================
ARUCO_AVAILABLE = hasattr(cv2, "aruco")

if ARUCO_AVAILABLE:
    ARUCO_DICTS = {
        "AprilTag_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
else:
    ARUCO_DICTS = {}


def detect_aruco(image_bgr, dict_id):
    """Detect ArUco markers using whichever OpenCV API is available."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=params
        )
    return corners, ids


# ============================================================
# Image canvas: shows image, captures clicks, draws overlays
# ============================================================
class ImageCanvas(QLabel):
    point_added = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(720, 480)
        self.setStyleSheet(
            "background-color: #1f1f1f; border: 1px solid #444; color: #888;"
        )
        self.setText("Load an image to begin")
        self.setFont(QFont("Arial", 12))

        self.original = None
        self.overlay = None
        self.disp_w = 0
        self.disp_h = 0
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0

        self.cal_points = []
        self.draw_enabled = False

    def set_image(self, bgr):
        self.original = bgr.copy()
        self.overlay = bgr.copy()
        self.cal_points = []
        self.refresh()

    def set_overlay(self, bgr):
        self.overlay = bgr.copy()
        self.refresh()

    def enable_drawing(self, on):
        self.draw_enabled = on
        if on and self.original is not None:
            self.cal_points = []
            self.overlay = self.original.copy()
            self.refresh()

    def clear_points(self):
        self.cal_points = []
        if self.original is not None:
            self.overlay = self.original.copy()
        self.refresh()

    def get_calibration_points(self):
        return list(self.cal_points)

    def refresh(self):
        if self.overlay is None:
            return

        img = self.overlay.copy()

        for pt in self.cal_points:
            cv2.circle(img, pt, 8, (0, 255, 255), -1)
            cv2.circle(img, pt, 8, (0, 0, 0), 2)
        if len(self.cal_points) == 2:
            cv2.line(img, self.cal_points[0], self.cal_points[1],
                     (0, 255, 255), 2)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)

        self.disp_w = scaled.width()
        self.disp_h = scaled.height()
        self.scale = scaled.width() / w if w else 1.0
        self.off_x = (self.width() - scaled.width()) // 2
        self.off_y = (self.height() - scaled.height()) // 2
        self.setPixmap(scaled)

    def mousePressEvent(self, event):
        if not self.draw_enabled or self.original is None:
            return
        if event.button() != Qt.LeftButton:
            return
        cx = event.x() - self.off_x
        cy = event.y() - self.off_y
        if cx < 0 or cy < 0 or cx > self.disp_w or cy > self.disp_h:
            return
        ox = int(round(cx / self.scale))
        oy = int(round(cy / self.scale))
        if len(self.cal_points) >= 2:
            self.cal_points = []
        self.cal_points.append((ox, oy))
        self.refresh()
        self.point_added.emit(len(self.cal_points))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()


# ============================================================
# Main application
# ============================================================
class CalibApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dimension Calibration & Measurement")
        self.resize(1150, 950)
        self.setStyleSheet("""
            QWidget { font-size: 12px; }
            QGroupBox { font-weight: bold; border: 1px solid #888;
                        border-radius: 4px; margin-top: 10px; padding-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { padding: 6px 14px; }
            QPushButton:disabled { color: #888; }
        """)

        self.image_bgr = None
        self.mm_per_px = None
        self.aruco_corners = None   # stored when ArUco calibration succeeds

        # Processing parameters driven by sliders
        self.threshold_offset = 0
        self.shadow_strength = 70
        self.remove_shadows = True
        self.clahe_enabled = True
        self.clahe_clip = 4.0
        self.use_otsu = True

        self._build_ui()
        self._wire_signals()

        if not ARUCO_AVAILABLE:
            self.r_aruco.setEnabled(False)
            self.r_aruco.setText("Method 2: ArUco (cv2.aruco not installed)")

    # ---------- UI ----------
    def _build_ui(self):
        method_box = QGroupBox("Calibration Method")
        self.r_ruler = QRadioButton("Method 1 — Ruler / Scale (manual line)")
        self.r_aruco = QRadioButton("Method 2 — ArUco Marker (auto-detect)")
        self.r_ruler.setChecked(True)
        m_lay = QHBoxLayout()
        m_lay.addWidget(self.r_ruler)
        m_lay.addWidget(self.r_aruco)
        m_lay.addStretch()
        method_box.setLayout(m_lay)

        top_row = QHBoxLayout()
        self.btn_load = QPushButton("📂 Load Image")
        self.lbl_status = QLabel("No image loaded")
        self.lbl_status.setStyleSheet("color: #555;")
        top_row.addWidget(self.btn_load)
        top_row.addWidget(self.lbl_status, 1)

        self.stack = QStackedWidget()

        # Method 1: ruler
        m1 = QWidget()
        m1_lay = QHBoxLayout()
        self.btn_draw_line = QPushButton("✏️ Draw Calibration Line (click 2 points)")
        self.btn_draw_line.setCheckable(True)
        self.spin_known_mm = QDoubleSpinBox()
        self.spin_known_mm.setRange(0.001, 100000.0)
        self.spin_known_mm.setDecimals(3)
        self.spin_known_mm.setValue(10.0)
        self.spin_known_mm.setSuffix(" mm")
        self.btn_calibrate_ruler = QPushButton("✅ Calibrate")
        self.btn_clear_line = QPushButton("Clear Points")
        m1_lay.addWidget(self.btn_draw_line)
        m1_lay.addWidget(QLabel("Known Distance:"))
        m1_lay.addWidget(self.spin_known_mm)
        m1_lay.addWidget(self.btn_calibrate_ruler)
        m1_lay.addWidget(self.btn_clear_line)
        m1_lay.addStretch()
        m1.setLayout(m1_lay)

        # Method 2: ArUco
        m2 = QWidget()
        m2_lay = QHBoxLayout()
        self.combo_dict = QComboBox()
        if ARUCO_DICTS:
            self.combo_dict.addItems(list(ARUCO_DICTS.keys()))
            self.combo_dict.setCurrentText("AprilTag_36h11")
        self.spin_marker_l = QDoubleSpinBox()
        self.spin_marker_l.setRange(0.01, 10000.0)
        self.spin_marker_l.setDecimals(3)
        self.spin_marker_l.setValue(30.0)
        self.spin_marker_l.setSuffix(" mm")
        self.spin_marker_b = QDoubleSpinBox()
        self.spin_marker_b.setRange(0.01, 10000.0)
        self.spin_marker_b.setDecimals(3)
        self.spin_marker_b.setValue(30.0)
        self.spin_marker_b.setSuffix(" mm")
        self.btn_detect_aruco = QPushButton("🔍 Detect Marker & Calibrate")
        m2_lay.addWidget(QLabel("Dictionary:"))
        m2_lay.addWidget(self.combo_dict)
        m2_lay.addWidget(QLabel("Length (L):"))
        m2_lay.addWidget(self.spin_marker_l)
        m2_lay.addWidget(QLabel("Breadth (B):"))
        m2_lay.addWidget(self.spin_marker_b)
        m2_lay.addWidget(self.btn_detect_aruco)
        m2_lay.addStretch()
        m2.setLayout(m2_lay)

        self.stack.addWidget(m1)
        self.stack.addWidget(m2)

        cal_row = QHBoxLayout()
        self.lbl_calib = QLabel("Calibration: not set")
        self.lbl_calib.setStyleSheet("color: #b00; font-weight: bold;")
        cal_row.addWidget(self.lbl_calib)
        cal_row.addStretch()

        # Measurement section with sliders
        meas_box = QGroupBox("Measurement")
        meas_root = QVBoxLayout()

        meas_btn_row = QHBoxLayout()
        self.btn_measure = QPushButton("📏 Detect & Measure OD / ID")
        self.btn_measure.setEnabled(False)
        self.btn_preview = QPushButton("🔍 Preview Filter")
        self.btn_preview.setEnabled(False)
        self.btn_show_orig = QPushButton("Show Original")
        meas_btn_row.addWidget(self.btn_measure)
        meas_btn_row.addWidget(self.btn_preview)
        meas_btn_row.addWidget(self.btn_show_orig)
        meas_btn_row.addStretch()

        # Shadow removal row
        shadow_row = QHBoxLayout()
        self.chk_shadow = QCheckBox("Remove Shadows")
        self.chk_shadow.setChecked(True)
        self.lbl_shadow_strength = QLabel("Shadow Strength: 70")
        self.sld_shadow = QSlider(Qt.Horizontal)
        self.sld_shadow.setRange(0, 100)
        self.sld_shadow.setValue(70)
        self.sld_shadow.setTickInterval(10)
        self.sld_shadow.setSingleStep(1)
        self.sld_shadow.setFixedWidth(200)
        shadow_row.addWidget(self.chk_shadow)
        shadow_row.addSpacing(16)
        shadow_row.addWidget(self.lbl_shadow_strength)
        shadow_row.addWidget(self.sld_shadow)
        shadow_row.addStretch()

        # CLAHE row
        clahe_row = QHBoxLayout()
        self.chk_clahe = QCheckBox("CLAHE")
        self.chk_clahe.setChecked(True)
        self.lbl_clahe_clip = QLabel("Clip Limit: 4.0")
        self.sld_clahe = QSlider(Qt.Horizontal)
        self.sld_clahe.setRange(5, 80)   # /10 → 0.5–8.0
        self.sld_clahe.setValue(40)
        self.sld_clahe.setTickInterval(5)
        self.sld_clahe.setSingleStep(1)
        self.sld_clahe.setFixedWidth(200)
        clahe_row.addWidget(self.chk_clahe)
        clahe_row.addSpacing(16)
        clahe_row.addWidget(self.lbl_clahe_clip)
        clahe_row.addWidget(self.sld_clahe)
        clahe_row.addStretch()

        # Threshold offset row
        thresh_row = QHBoxLayout()
        self.chk_otsu = QCheckBox("Otsu")
        self.chk_otsu.setChecked(True)
        self.lbl_thresh = QLabel("Threshold Adjust:  0")
        self.sld_thresh = QSlider(Qt.Horizontal)
        self.sld_thresh.setRange(-80, 80)
        self.sld_thresh.setValue(0)
        self.sld_thresh.setTickInterval(10)
        self.sld_thresh.setSingleStep(1)
        self.sld_thresh.setFixedWidth(200)
        thresh_row.addWidget(self.chk_otsu)
        thresh_row.addSpacing(16)
        thresh_row.addWidget(self.lbl_thresh)
        thresh_row.addWidget(self.sld_thresh)
        thresh_row.addStretch()

        meas_root.addLayout(meas_btn_row)
        meas_root.addLayout(shadow_row)
        meas_root.addLayout(clahe_row)
        meas_root.addLayout(thresh_row)
        meas_box.setLayout(meas_root)

        self.canvas = ImageCanvas()

        self.txt_result = QPlainTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMaximumHeight(130)
        self.txt_result.setStyleSheet(
            "background:#111; color:#4dff88; "
            "font-family: Consolas, 'Courier New', monospace;"
        )
        self.txt_result.setPlainText(
            "1) Load an image.  2) Pick a method & calibrate.  3) Click "
            "'Detect & Measure OD / ID'."
        )

        root = QVBoxLayout()
        root.addWidget(method_box)
        root.addLayout(top_row)
        root.addWidget(self.stack)
        root.addLayout(cal_row)
        root.addWidget(meas_box)
        root.addWidget(self.canvas, 1)
        root.addWidget(self.txt_result)
        self.setLayout(root)

    def _wire_signals(self):
        self.r_ruler.toggled.connect(
            lambda c: self.stack.setCurrentIndex(0) if c else None
        )
        self.r_aruco.toggled.connect(
            lambda c: self.stack.setCurrentIndex(1) if c else None
        )
        self.btn_load.clicked.connect(self.load_image)
        self.btn_draw_line.toggled.connect(self.on_draw_toggle)
        self.btn_clear_line.clicked.connect(self.canvas.clear_points)
        self.btn_calibrate_ruler.clicked.connect(self.calibrate_from_ruler)
        self.btn_detect_aruco.clicked.connect(self.calibrate_from_aruco)
        self.btn_measure.clicked.connect(self.measure_object)
        self.btn_preview.clicked.connect(self.show_filter_preview)
        self.btn_show_orig.clicked.connect(self.show_original)
        self.canvas.point_added.connect(self.on_point_added)

        self.sld_thresh.valueChanged.connect(self._on_thresh_changed)
        self.sld_shadow.valueChanged.connect(self._on_shadow_strength_changed)
        self.chk_shadow.stateChanged.connect(self._on_shadow_toggle)
        self.sld_clahe.valueChanged.connect(self._on_clahe_clip_changed)
        self.chk_clahe.stateChanged.connect(self._on_clahe_toggle)
        self.chk_otsu.stateChanged.connect(self._on_otsu_toggle)

    # ---------- Slider slots ----------
    def _on_thresh_changed(self, value):
        self.threshold_offset = value
        self.lbl_thresh.setText(f"Threshold Adjust: {value:+d}")
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    def _on_shadow_strength_changed(self, value):
        self.shadow_strength = value
        self.lbl_shadow_strength.setText(f"Shadow Strength: {value}")
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    def _on_shadow_toggle(self, state):
        self.remove_shadows = (state == Qt.Checked)
        self.sld_shadow.setEnabled(self.remove_shadows)
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    def _on_clahe_clip_changed(self, value):
        self.clahe_clip = value / 10.0
        self.lbl_clahe_clip.setText(f"Clip Limit: {self.clahe_clip:.1f}")
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    def _on_clahe_toggle(self, state):
        self.clahe_enabled = (state == Qt.Checked)
        self.sld_clahe.setEnabled(self.clahe_enabled)
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    def _on_otsu_toggle(self, state):
        self.use_otsu = (state == Qt.Checked)
        if self.image_bgr is not None and self.btn_preview.isEnabled():
            self.show_filter_preview()

    # ---------- Slots ----------
    def show_original(self):
        if self.image_bgr is not None:
            self.canvas.set_image(self.image_bgr)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to load image.")
            return
        self.image_bgr = img
        self.aruco_corners = None
        self.canvas.set_image(img)
        self.lbl_status.setText(
            f"Loaded: {os.path.basename(path)}  "
            f"({img.shape[1]}×{img.shape[0]})"
        )
        self.btn_measure.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.mm_per_px = None
        self.lbl_calib.setText("Calibration: not set")
        self.lbl_calib.setStyleSheet("color: #b00; font-weight: bold;")
        self.txt_result.setPlainText(
            "Image loaded. Pick a calibration method, then calibrate."
        )

    def on_draw_toggle(self, checked):
        self.canvas.enable_drawing(checked)
        if checked:
            self.btn_draw_line.setText("✏️ Drawing… click 2 points on ruler")
            self.txt_result.setPlainText(
                "Click TWO points on the ruler/scale that are a known "
                "distance apart (e.g. the 0 mm and 10 mm tick marks)."
            )
        else:
            self.btn_draw_line.setText(
                "✏️ Draw Calibration Line (click 2 points)"
            )

    def on_point_added(self, n):
        if n == 1:
            self.txt_result.setPlainText(
                "First point set. Click the second point on the ruler."
            )
        elif n == 2:
            p1, p2 = self.canvas.get_calibration_points()
            dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
            self.txt_result.setPlainText(
                "Two points selected.\n"
                f"  Point 1: ({p1[0]}, {p1[1]})\n"
                f"  Point 2: ({p2[0]}, {p2[1]})\n"
                f"  Pixel distance: {dist:.2f} px\n"
                "Set the known distance (mm) and click Calibrate."
            )

    # ---------- Method 1: Ruler ----------
    def calibrate_from_ruler(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        pts = self.canvas.get_calibration_points()
        if len(pts) != 2:
            QMessageBox.warning(
                self, "Need 2 points",
                "Click 'Draw Calibration Line', then click 2 points on the ruler."
            )
            return
        p1, p2 = pts
        pixel_dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        if pixel_dist < 1.0:
            QMessageBox.warning(self, "Too close",
                                "The two points are essentially the same.")
            return

        known_mm = self.spin_known_mm.value()
        self.mm_per_px = known_mm / pixel_dist
        self._set_calibrated(
            f"✅ Calibrated via ruler.\n"
            f"  Pixel distance:  {pixel_dist:.3f} px\n"
            f"  Known distance:  {known_mm:.3f} mm\n"
            f"  Scale factor:    {self.mm_per_px:.6f} mm/px\n"
            "You can now click 'Detect & Measure OD / ID'."
        )
        self.btn_draw_line.setChecked(False)
        self.canvas.enable_drawing(False)

    # ---------- Method 2: ArUco ----------
    def calibrate_from_aruco(self):
        if self.image_bgr is None:
            QMessageBox.warning(self, "No image", "Load an image first.")
            return
        if not ARUCO_AVAILABLE:
            QMessageBox.critical(
                self, "ArUco missing",
                "cv2.aruco isn't available.\n"
                "Install opencv-contrib-python:\n"
                "    pip install opencv-contrib-python"
            )
            return

        dict_id = ARUCO_DICTS[self.combo_dict.currentText()]
        try:
            corners, ids = detect_aruco(self.image_bgr, dict_id)
        except Exception as e:
            QMessageBox.critical(self, "ArUco Error",
                                 f"Detection failed:\n{e}")
            return

        if ids is None or len(corners) == 0:
            QMessageBox.warning(
                self, "Not found",
                "No ArUco marker detected.\n"
                "Try a different dictionary, better lighting, or "
                "make sure the marker isn't blurred or cropped."
            )
            return

        # Store all detected marker corners for exclusion during measurement
        self.aruco_corners = corners

        marker = corners[0][0]
        side_top    = np.linalg.norm(marker[0] - marker[1])
        side_right  = np.linalg.norm(marker[1] - marker[2])
        side_bottom = np.linalg.norm(marker[2] - marker[3])
        side_left   = np.linalg.norm(marker[3] - marker[0])

        avg_horiz = (side_top + side_bottom) / 2.0
        avg_vert  = (side_left + side_right) / 2.0

        L_mm = self.spin_marker_l.value()
        B_mm = self.spin_marker_b.value()
        mm_per_px_h = L_mm / avg_horiz
        mm_per_px_v = B_mm / avg_vert
        self.mm_per_px = (mm_per_px_h + mm_per_px_v) / 2.0

        overlay = self.image_bgr.copy()
        int_corners = [c.astype(np.int32) for c in corners]
        cv2.polylines(overlay, int_corners, True, (0, 255, 0), 3)
        cx, cy = np.mean(marker, axis=0).astype(int)
        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(overlay, f"ID:{int(ids[0][0])}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        self.canvas.set_overlay(overlay)

        self._set_calibrated(
            f"✅ Calibrated via ArUco marker (ID {int(ids[0][0])}).\n"
            f"  Marker size entered:   {L_mm} mm × {B_mm} mm\n"
            f"  Detected horizontal:   {avg_horiz:.2f} px  ->  "
            f"{mm_per_px_h:.6f} mm/px\n"
            f"  Detected vertical:     {avg_vert:.2f} px  ->  "
            f"{mm_per_px_v:.6f} mm/px\n"
            f"  Average scale factor:  {self.mm_per_px:.6f} mm/px\n"
            "ArUco region will be excluded from OD/ID detection.\n"
            "You can now click 'Detect & Measure OD / ID'."
        )

    def _set_calibrated(self, msg):
        self.lbl_calib.setText(
            f"Calibration: {self.mm_per_px:.6f} mm/px   "
            f"({1 / self.mm_per_px:.3f} px/mm)"
        )
        self.lbl_calib.setStyleSheet("color: #060; font-weight: bold;")
        self.btn_measure.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.txt_result.setPlainText(msg)

    # ---------- Filter preview ----------
    def show_filter_preview(self):
        if self.image_bgr is None:
            return

        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        corrected = self._preprocess_gray(gray)

        if self.use_otsu:
            otsu_val, _ = cv2.threshold(
                corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            adjusted_val = int(np.clip(otsu_val + self.threshold_offset, 0, 255))
        else:
            otsu_val = None
            adjusted_val = int(np.clip(127 + self.threshold_offset, 0, 255))
        _, thresh = cv2.threshold(corrected, adjusted_val, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Show ArUco exclusion zone in gray on the preview
        preview_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        if self.aruco_corners is not None:
            for corner_arr in self.aruco_corners:
                pts = corner_arr[0].astype(np.float32)
                cx_m = int(pts[:, 0].mean())
                cy_m = int(pts[:, 1].mean())
                padded = pts.copy()
                padded[:, 0] = cx_m + (padded[:, 0] - cx_m) * 1.3
                padded[:, 1] = cy_m + (padded[:, 1] - cy_m) * 1.3
                cv2.fillPoly(preview_bgr, [padded.astype(np.int32)], (80, 80, 200))
                cv2.polylines(preview_bgr, [corner_arr[0].astype(np.int32)],
                              True, (0, 220, 220), 2)

        self.canvas.set_overlay(preview_bgr)
        otsu_str = f"Otsu={otsu_val}" if otsu_val is not None else f"manual={adjusted_val}"
        clahe_str = f"CLAHE clip={self.clahe_clip:.1f}" if self.clahe_enabled else "CLAHE off"
        self.txt_result.setPlainText(
            f"Filter preview  ({otsu_str}  offset={self.threshold_offset:+d}  "
            f"shadow={'on' if self.remove_shadows else 'off'} str={self.shadow_strength}  {clahe_str})\n"
            "Blue region = ArUco tag (excluded from measurement).\n"
            "Adjust sliders until the object boundary is clean, then click 'Detect & Measure OD / ID'."
        )

    # ---------- Measurement ----------
    def measure_object(self):
        if self.image_bgr is None or self.mm_per_px is None:
            return

        gray = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2GRAY)
        corrected = self._preprocess_gray(gray)

        if self.use_otsu:
            otsu_val, _ = cv2.threshold(
                corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            adjusted_val = int(np.clip(otsu_val + self.threshold_offset, 0, 255))
        else:
            adjusted_val = int(np.clip(127 + self.threshold_offset, 0, 255))
        _, thresh = cv2.threshold(corrected, adjusted_val, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Erase the ArUco tag region(s) so they are never picked as OD/ID
        if self.aruco_corners is not None:
            for corner_arr in self.aruco_corners:
                pts = corner_arr[0].astype(np.int32)
                # add a small padding (10% of marker size) around each marker
                cx_m = int(pts[:, 0].mean())
                cy_m = int(pts[:, 1].mean())
                padded = pts.copy().astype(np.float32)
                padded[:, 0] = cx_m + (padded[:, 0] - cx_m) * 1.3
                padded[:, 1] = cy_m + (padded[:, 1] - cy_m) * 1.3
                cv2.fillPoly(thresh, [padded.astype(np.int32)], 0)

        clean = self._clean_otsu_mask(thresh)
        outer_measure, inner_measure = self._detect_contour_diameters(clean)

        if outer_measure is None:
            QMessageBox.information(
                self, "No object",
                "No contours found outside the ArUco region.\n"
                "Try adjusting the threshold or shadow sliders."
            )
            return

        out = self.image_bgr.copy()
        cv2.drawContours(out, [outer_measure["contour"]], -1, (0, 255, 0), 2)
        self._draw_contour_fit(out, outer_measure, (0, 255, 0))

        od_px = outer_measure["diameter"]
        od_mm = od_px * self.mm_per_px
        msg = f"Outer Diameter (OD): {od_mm:.3f} mm   ({od_px:.1f} px)\n"

        if inner_measure is not None:
            cv2.drawContours(out, [inner_measure["contour"]], -1, (255, 0, 0), 2)
            self._draw_contour_fit(out, inner_measure, (255, 0, 0))
            id_px = inner_measure["diameter"]
            id_mm = id_px * self.mm_per_px
            msg += (f"Inner Diameter (ID): {id_mm:.3f} mm   ({id_px:.1f} px)\n"
                    f"Wall thickness:      {(od_mm - id_mm) / 2:.3f} mm")
        else:
            msg += "Inner Diameter (ID): not detected"

        # Annotate image
        cnt_pts = outer_measure["contour"].reshape(-1, 2)
        cx, cy = np.mean(cnt_pts, axis=0).astype(int)
        cv2.putText(out, f"OD={od_mm:.2f}mm",
                    (max(10, cx - 90), max(20, cy - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if inner_measure is not None:
            cv2.putText(out, f"ID={id_mm:.2f}mm",
                        (max(10, cx - 90), cy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Mark excluded ArUco regions in cyan on the output
        if self.aruco_corners is not None:
            int_c = [c.astype(np.int32) for c in self.aruco_corners]
            cv2.polylines(out, int_c, True, (255, 255, 0), 2)

        self.canvas.set_overlay(out)
        clahe_info = f"CLAHE clip={self.clahe_clip:.1f}" if self.clahe_enabled else "CLAHE off"
        otsu_info = "Otsu" if self.use_otsu else "manual"
        self.txt_result.setPlainText(
            f"✅ Measurement  (scale {self.mm_per_px:.6f} mm/px  |  "
            f"thresh {otsu_info} offset {self.threshold_offset:+d}  |  "
            f"shadow {'on' if self.remove_shadows else 'off'} str={self.shadow_strength}  |  "
            f"{clahe_info})\n"
            f"{msg}"
        )

    # ---------- Processing helpers (ported from ui_otsu (1).py) ----------
    def _preprocess_gray(self, gray):
        if self.remove_shadows:
            result = self._normalize_illumination(gray)
        else:
            result = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.clahe_enabled:
            clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
            result = clahe.apply(result)
        return result

    def _normalize_illumination(self, gray):
        image_h, image_w = gray.shape[:2]
        min_size = min(image_w, image_h)
        bg_size = self._odd_kernel(min_size * 0.22, 41)
        sh_size = self._odd_kernel(min_size * 0.08, 15)
        strength = self.shadow_strength / 100.0

        background = cv2.GaussianBlur(gray, (bg_size, bg_size), 0)
        divided = cv2.divide(gray, background, scale=255)

        sh_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (sh_size, sh_size)
        )
        dark_halo   = cv2.morphologyEx(divided, cv2.MORPH_BLACKHAT, sh_kernel)
        bright_halo = cv2.morphologyEx(divided, cv2.MORPH_TOPHAT, sh_kernel)
        corrected = cv2.addWeighted(divided, 1.0, dark_halo, strength, 0)
        corrected = cv2.addWeighted(corrected, 1.0, bright_halo, -0.35 * strength, 0)

        return cv2.GaussianBlur(corrected, (5, 5), 0)

    def _clean_otsu_mask(self, mask):
        image_h, image_w = mask.shape[:2]
        min_size = min(image_w, image_h)
        cleaned = np.zeros_like(mask)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        min_area = max(30, int(min_size * min_size * 0.0005))

        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            touches_border = (
                x <= 1 or y <= 1 or x + w >= image_w - 1 or y + h >= image_h - 1
            )
            too_large = w > image_w * 0.75 or h > image_h * 0.75
            if area >= min_area and not touches_border and not too_large:
                cleaned[labels == label] = 255

        k = self._odd_kernel(min_size * 0.018, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
        return cleaned

    def _detect_contour_diameters(self, mask):
        image_h, image_w = mask.shape[:2]
        min_size = min(image_w, image_h)
        k = self._odd_kernel(min_size * 0.025, 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours or hierarchy is None:
            return None, None

        hierarchy = hierarchy[0]
        best_outer = None
        best_inner = None
        best_score = -1

        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area <= 0 or perimeter <= 0:
                continue
            _, _, w, h = cv2.boundingRect(contour)
            if w < min_size * 0.03 or h < min_size * 0.03:
                continue
            if w > image_w * 0.80 or h > image_h * 0.80:
                continue

            children = [
                ci for ci, item in enumerate(hierarchy)
                if item[3] == index and cv2.contourArea(contours[ci]) > 0
            ]
            if not children:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.25:
                continue

            child = max(children, key=lambda ci: cv2.contourArea(contours[ci]))
            child_area = cv2.contourArea(contours[child])
            if child_area < area * 0.05:
                continue

            score = area + child_area + circularity * 5000
            if score > best_score:
                best_outer = contour
                best_inner = contours[child]
                best_score = score

        if best_outer is None:
            best_outer = self._find_largest_outer(contours, hierarchy, image_w, image_h)
            if best_outer is not None:
                best_inner = self._find_inner_from_filled(connected, best_outer)

        if best_outer is None:
            return None, None

        outer_m = self._measure_contour(best_outer)
        inner_m = self._measure_contour(best_inner) if best_inner is not None else None

        if (outer_m is not None and inner_m is not None
                and inner_m["diameter"] >= outer_m["diameter"]):
            inner_m = None

        return outer_m, inner_m

    def _find_largest_outer(self, contours, hierarchy, image_w, image_h):
        best, best_area = None, 0
        for index, contour in enumerate(contours):
            if hierarchy[index][3] != -1:
                continue
            area = cv2.contourArea(contour)
            _, _, w, h = cv2.boundingRect(contour)
            if area > best_area and w < image_w * 0.80 and h < image_h * 0.80:
                best, best_area = contour, area
        return best

    def _find_inner_from_filled(self, mask, outer_contour):
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, [outer_contour], -1, 255, -1)
        hole_mask = cv2.subtract(filled, mask)
        hole_mask = cv2.morphologyEx(
            hole_mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        )
        hole_contours, _ = cv2.findContours(
            hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        min_hole = cv2.contourArea(outer_contour) * 0.03
        valid = [c for c in hole_contours if cv2.contourArea(c) > min_hole]
        return max(valid, key=cv2.contourArea) if valid else None

    def _measure_contour(self, contour):
        if contour is None or len(contour) == 0:
            return None
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (a, b), _ = ellipse
            return {"contour": contour, "diameter": float((a + b) / 2), "ellipse": ellipse}
        _, _, w, h = cv2.boundingRect(contour)
        return {"contour": contour, "diameter": float((w + h) / 2), "ellipse": None}

    def _draw_contour_fit(self, output, measurement, color):
        if measurement["ellipse"] is not None:
            cv2.ellipse(output, measurement["ellipse"], color, 1)

    def _odd_kernel(self, value, minimum):
        size = max(minimum, int(round(value)))
        return size + 1 if size % 2 == 0 else size


# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = CalibApp()
    w.show()
    sys.exit(app.exec_())
