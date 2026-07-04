import sys
import onnxruntime  # Preload before PyQt5 to avoid Windows DLL initialization conflicts.
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout,
    QSlider, QGroupBox, QFormLayout, QScrollArea,
    QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from ultralytics import YOLO


class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Removal & Bangle Extraction Pipeline")
        self.setGeometry(100, 100, 1400, 800)

        # Load models
        print("Loading handremove.onnx model...")
        self.hand_model = YOLO(r"N:\Projects\handdataset\handremove.onnx", task="segment")
        print("Models loaded successfully!")

        self.original_image = None
        self.step_images = {}  # Store images from each step for debugging

        # UI Setup
        self.init_ui()

    def init_ui(self):
        # Create scroll area for step-by-step images
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_content = QWidget()
        self.step_grid = QGridLayout(scroll_content)
        
        # Image display labels - all steps for debugging
        self.step_labels = {}
        step_names = [
            ("Original", "Original Image"),
            ("Step1_BG", "Step 1: White BG Removed"),
            ("Step2_Shadow", "Step 2: Shadows Removed"),
            ("Step3_Hand_YOLO", "Step 3: Hand Removed (YOLO)"),
            ("Step4_HSV", "Step 4: HSV Color Filter"),
            ("Step5_Morphology", "Step 5: Morphology Ops"),
            ("Step6_Final", "Step 6: Final Output")
        ]
        
        for i, (key, name) in enumerate(step_names):
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("border: 2px solid gray; background-color: #f0f0f0; min-height: 200px; min-width: 200px;")
            self.step_grid.addWidget(label, i // 4, (i % 4) * 2, 1, 2)
            
            info_label = QLabel("Awaiting processing...")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
            self.step_grid.addWidget(info_label, (i // 4) + 1, (i % 4) * 2, 1, 2)
            
            self.step_labels[key] = (label, info_label)
        
        scroll.setWidget(scroll_content)

        # Control panel
        self.load_btn = QPushButton("Load Image")
        self.process_btn = QPushButton("Process Image")
        self.process_btn.setEnabled(False)

        self.load_btn.clicked.connect(self.load_image)
        self.process_btn.clicked.connect(self.process_image)

        # Sliders for Tuning
        self.sat_slider = QSlider(Qt.Horizontal)
        self.sat_slider.setRange(0, 255)
        self.sat_slider.setValue(80)
        self.sat_slider.valueChanged.connect(self.on_slider_change)

        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(100, 5000)
        self.min_area_slider.setValue(500)
        self.min_area_slider.valueChanged.connect(self.on_slider_change)

        slider_layout = QFormLayout()
        slider_layout.addRow("Min Saturation:", self.sat_slider)
        slider_layout.addRow("Min Contour Area:", self.min_area_slider)
        
        slider_group = QGroupBox("HSV Extraction Tuning")
        slider_group.setLayout(slider_layout)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.process_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll, 1)
        main_layout.addWidget(slider_group)
        main_layout.addLayout(button_layout)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.setLayout(main_layout)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                QMessageBox.critical(self, "Error", "Could not load image!")
                return
            
            self.step_images = {}  # Clear previous steps
            self.show_step_image("Original", self.original_image, 
                               f"Loaded: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
            self.process_btn.setEnabled(True)
            print(f"Image loaded: {file_path}")

    def process_image(self):
        if self.original_image is None:
            return

        print("\n" + "="*50)
        print("Starting processing pipeline...")
        print("="*50)
        
        bgr = self.original_image.copy()
        
        # ── Step 1: Remove white/gray background ──────────────────────────────
        print("\n[Step 1] Removing white background...")
        mask = self._remove_white_bg(bgr)
        self._log_step("Step 1 - White BG", mask)
        step1_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step1_BG", step1_output, self._get_mask_stats(mask))

        # ── Step 2: Remove shadows (hand and gold shadows) ─────────────────────
        print("\n[Step 2] Removing shadows...")
        mask = self._remove_shadows(bgr, mask)
        self._log_step("Step 2 - Shadows Removed", mask)
        step2_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step2_Shadow", step2_output, self._get_mask_stats(mask))

        # ── Step 3: Remove bright green plastic clip ──────────────────────────
        print("\n[Step 3] Removing green clip...")
        mask = self._remove_green(bgr, mask)
        self._log_step("Step 3 - No Green", mask)
        step3_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step3_Green", step3_output, self._get_mask_stats(mask))

        # ── Step 3: Remove human hand using YOLO (handremove.onnx) ─────────────
        print("\n[Step 3] Removing hand using YOLO (handremove.onnx)...")
        mask = self._remove_hand_yolo(bgr, mask)
        self._log_step("Step 3 - No Hand (YOLO)", mask)
        step3_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step3_Hand_YOLO", step3_output, self._get_mask_stats(mask))

        # ── Step 4: Keep only colorful (high-saturation) pixels ───────────────
        print("\n[Step 4] Applying HSV color filter...")
        min_sat = self.sat_slider.value()
        mask = self._keep_colorful(bgr, mask, min_sat=min_sat)
        self._log_step("Step 4 - HSV Colorful", mask)
        step4_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step4_HSV", step4_output, self._get_mask_stats(mask))

        # ── Step 5: Close ring gaps, remove tiny fragments ────────────────────
        print("\n[Step 5] Morphological operations...")
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
        min_area = self.min_area_slider.value()
        mask = self._remove_small(mask, min_area=min_area)
        self._log_step("Step 5 - Morphology", mask)
        step5_output = self._apply_mask_to_white(bgr, mask)
        self.show_step_image("Step5_Morphology", step5_output, self._get_mask_stats(mask))

        # ── Step 6: Final output ──────────────────────────────────────────────
        print("\n[Step 6] Generating final output...")
        white_bg = np.full_like(bgr, 255)
        final = np.where(mask[:, :, None] == 255, bgr, white_bg)
        self._log_step("Step 6 - Final", mask)
        self.show_step_image("Step6_Final", final, self._get_mask_stats(mask))

        if not np.any(mask):
            print("\n⚠️  WARNING: Result is empty — check debug steps for which step empties the mask.")
            QMessageBox.warning(self, "Warning", 
                "Result is empty! Check the step-by-step images to see where the mask was lost.")

        print("\n" + "="*50)
        print("Processing complete!")
        print("="*50)

    # ── Processing Functions ──────────────────────────────────────────────────

    def _remove_white_bg(self, bgr):
        """Keep pixels darker than 215 (gray). Background is white/light gray >230."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def _remove_shadows(self, bgr, mask):
        """
        Remove hand shadows and gold shadows using LAB color space and brightness analysis.
        Detects dark areas (shadows) while preserving the bangle.
        """
        print("  Analyzing shadows...")
        # Convert to LAB color space for better shadow detection
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Lightness channel
        
        # Detect VERY dark areas (shadows) - only L < 40 (much stricter)
        shadow_mask = cv2.inRange(l_channel, 0, 40)
        
        # Also detect areas with very low saturation and extremely low brightness
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Only catch pixels that are truly desaturated AND very dark
        gray_shadow_mask = cv2.inRange(saturation, 0, 30) & cv2.inRange(value, 0, 40)
        
        # Combine shadow masks
        combined_shadow = cv2.bitwise_or(shadow_mask, gray_shadow_mask)
        
        # Smaller dilation - don't expand shadows too much
        kernel = np.ones((5, 5), np.uint8)
        expanded_shadow = cv2.dilate(combined_shadow, kernel, iterations=1)
        
        # Remove shadows from mask
        shadow_removed_mask = cv2.bitwise_and(mask, cv2.bitwise_not(expanded_shadow))
        
        shadow_px = int(np.sum(expanded_shadow > 0))
        print(f"  Shadows detected: {shadow_px:,} pixels")
        
        return shadow_removed_mask

    def _remove_green(self, bgr, mask):
        """Remove bright green plastic clip."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green = cv2.inRange(hsv, np.array([45, 80, 80]), np.array([85, 255, 255]))
        green = cv2.dilate(green, np.ones((9, 9), np.uint8), iterations=1)
        return cv2.bitwise_and(mask, cv2.bitwise_not(green))

    def _remove_hand_yolo(self, bgr, mask):
        """Improved hand removal with better coverage"""
        print("  Running YOLO segmentation with handremove.onnx...")
        results = self.hand_model(bgr, imgsz=640, conf=0.25, device="cpu", verbose=False)  # ← increased imgsz + lowered conf
        result = results[0]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            combined_mask = np.zeros(bgr.shape[:2], dtype=np.uint8)

            for m in masks:
                m = cv2.resize(m, (bgr.shape[1], bgr.shape[0]))
                m = (m > 0.4).astype(np.uint8) * 255   # ← lowered threshold
                combined_mask = cv2.bitwise_or(combined_mask, m)

            # Much stronger dilation + extra iterations
            hand_kernel = np.ones((25, 25), np.uint8)   # ← bigger kernel
            expanded_hand_mask = cv2.dilate(combined_mask, hand_kernel, iterations=2)

            # Optional: Add morphological closing to fill holes in hand detection
            expanded_hand_mask = cv2.morphologyEx(expanded_hand_mask, cv2.MORPH_CLOSE, 
                                                np.ones((15, 15), np.uint8))

            px = int(np.sum(expanded_hand_mask > 0))
            print(f"  Hand detected: {px:,} pixels")

            if px > 800:   # ← raised threshold
                return cv2.bitwise_and(mask, cv2.bitwise_not(expanded_hand_mask))
            else:
                print("  No significant hand found")
        else:
            print("  No masks detected by YOLO")

        return mask
    def _keep_colorful(self, bgr, mask, min_sat=80):
        """
        Retain only pixels whose HSV saturation >= min_sat.
        Removes grayscale artifacts (QR code, B&W labels, shadows) while
        preserving vivid gold and gemstone colors.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        colorful = (hsv[:, :, 1] >= min_sat).astype(np.uint8) * 255
        # Dilate so bangle edges (slightly desaturated due to lighting) are included
        colorful = cv2.dilate(colorful, np.ones((9, 9), np.uint8), iterations=1)
        return cv2.bitwise_and(mask, colorful)

    def _remove_small(self, mask, min_area=500):
        """Remove small contours below min_area threshold."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.zeros_like(mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                cv2.drawContours(out, [cnt], -1, 255, -1)
        return out

    # ── Utility Functions ─────────────────────────────────────────────────────

    def _apply_mask_to_white(self, bgr, mask):
        """Apply mask to image with white background."""
        white_bg = np.full_like(bgr, 255)
        return np.where(mask[:, :, None] == 255, bgr, white_bg)

    def _get_mask_stats(self, mask):
        """Get statistics about the mask."""
        px = int(np.sum(mask > 0))
        percentage = 100 * px / mask.size if mask.size > 0 else 0
        return f"Mask: {px:,} px ({percentage:.1f}%)"

    def _log_step(self, label, mask):
        """Log step information to console."""
        stats = self._get_mask_stats(mask)
        print(f"  {label}: {stats}")

    def show_step_image(self, step_key, image, info_text):
        """Display an image in the corresponding step label."""
        if step_key in self.step_labels:
            label, info_label = self.step_labels[step_key]
            self._set_label_image(label, image)
            info_label.setText(info_text)
            info_label.setStyleSheet("color: black; font-weight: bold;")
            self.step_images[step_key] = image

    def _set_label_image(self, label, image):
        """Convert OpenCV image to QPixmap and set it on a QLabel."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        qt_image = QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio
            )
        )

    def on_slider_change(self):
        """Re-process image when sliders change (if already processed)."""
        if self.original_image is not None and self.process_btn.isEnabled():
            # Auto-reprocess when sliders change
            self.process_image()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    window = SegmentationApp()
    window.show()

    sys.exit(app.exec_())
