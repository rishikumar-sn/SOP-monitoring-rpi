#!/usr/bin/env python3
import sys
import os
import time
import math
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from hailo_platform import VDevice, FormatType


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HEF_PATH = os.path.join(SCRIPT_DIR, "real_esrgan_x2.hef")
M = 512
PAD = 64
STRIDE = M - 2 * PAD


def numpy_to_qpixmap(arr):
    h, w = arr.shape[:2]
    img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(
        QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888))


def tile_grid(w, h):
    cols = max(1, math.ceil((w - STRIDE) / STRIDE) + 1)
    rows = max(1, math.ceil((h - STRIDE) / STRIDE) + 1)
    return cols, rows


class SuperResolutionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-ESRGAN x2 (Hailo)")
        self.setMinimumSize(1200, 750)

        self.source_image = None
        self.source_path = None
        self.result_image = None
        self.running = False

        self.vdevice = None
        self.model = None
        self.cmodel = None
        self.bindings = None
        self._in_buf = np.zeros((M, M, 3), dtype=np.uint8)
        self._out_buf = np.zeros((1024, 1024, 3), dtype=np.uint8)

        self._init_ui()
        self._load_model()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        btn_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.run_btn = QPushButton("Run Super Resolution")
        self.save_btn = QPushButton("Save Result")
        self.run_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.load_btn.clicked.connect(self._load_image)
        self.run_btn.clicked.connect(self._run)
        self.save_btn.clicked.connect(self._save_result)
        btn_row.addWidget(self.load_btn)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.save_btn)
        layout.addLayout(btn_row)

        info_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.lbl_time = QLabel("")
        self.lbl_tile = QLabel("")
        info_row.addWidget(self.progress, 1)
        info_row.addWidget(self.lbl_time)
        info_row.addWidget(self.lbl_tile)
        layout.addLayout(info_row)

        img_row = QHBoxLayout()
        g1 = QGroupBox("Original")
        v1 = QVBoxLayout(g1)
        self.orig_lbl = QLabel("No image loaded")
        self.orig_lbl.setAlignment(Qt.AlignCenter)
        self.orig_lbl.setMinimumSize(300, 300)
        self.orig_lbl.setStyleSheet("border:1px solid gray")
        v1.addWidget(self.orig_lbl)
        img_row.addWidget(g1)

        g2 = QGroupBox("Super Resolved (2x)")
        v2 = QVBoxLayout(g2)
        self.result_lbl = QLabel("Run inference to see result")
        self.result_lbl.setAlignment(Qt.AlignCenter)
        self.result_lbl.setMinimumSize(300, 300)
        self.result_lbl.setStyleSheet("border:1px solid gray")
        v2.addWidget(self.result_lbl)
        img_row.addWidget(g2)
        layout.addLayout(img_row)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

    def _load_model(self):
        try:
            self.status.setText("Loading HEF model…")
            QApplication.processEvents()
            self.vdevice = VDevice()
            self.model = self.vdevice.create_infer_model(HEF_PATH)
            self.model.input().set_format_type(FormatType.UINT8)
            self.cmodel = self.model.configure()
            self.bindings = self.cmodel.create_bindings()
            self.bindings.output().set_buffer(self._out_buf)
            self.status.setText("Model loaded")
        except Exception as e:
            QMessageBox.critical(self, "Model Load Error",
                                 f"Could not load model.\n"
                                 f"Make sure the Hailo device is available.\n\n{e}")
            self.status.setText("Model failed to load")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.tif)")
        if not path:
            return
        self.source_path = path
        self.source_image = cv2.imread(path)
        if self.source_image is None:
            QMessageBox.warning(self, "Error", "Could not load image")
            return
        h, w = self.source_image.shape[:2]
        self._show(self.source_image, self.orig_lbl)
        self.status.setText(f"{os.path.basename(path)} ({w}x{h})")
        self.run_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.result_image = None
        self.result_lbl.setText("Run inference to see result")
        self.lbl_time.setText("")
        self.lbl_tile.setText("")

    def _show(self, cv_img, label, mx=500):
        h, w = cv_img.shape[:2]
        s = min(mx / w, mx / h)
        if s < 1:
            d = cv2.resize(cv_img, (int(w * s), int(h * s)),
                           interpolation=cv2.INTER_AREA)
        else:
            d = cv_img.copy()
        label.setPixmap(numpy_to_qpixmap(d))

    # ---- inference helpers -------------------------------------------------

    def _infer(self, tile_rgb):
        """Run 512x512 RGB tile on Hailo.  Device must be activated."""
        self._in_buf[:] = tile_rgb
        self.bindings.input().set_buffer(self._in_buf)
        self.cmodel.run([self.bindings], timeout=60000)
        raw = self.bindings.output().get_buffer()
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    def _run_single(self, img):
        """Image fits in 512×512 — letterbox, infer, un-letterbox."""
        h, w = img.shape[:2]
        s = min(M / w, M / h)
        nw, nh = int(w * s), int(h * s)
        r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (M - nh) // 2, (M - nw) // 2
        canvas = np.zeros((M, M, 3), dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = r
        out = self._infer(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        sy, sx = int(top * 2), int(left * 2)
        crop = out[sy:sy + nh * 2, sx:sx + nw * 2]
        return cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

    def _run_tiled(self, img, on_tile):
        """Image larger than 512×512 — tile with overlap context."""
        h, w = img.shape[:2]
        nc, nr = tile_grid(w, h)
        total = nc * nr
        out = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        idx = 0

        for row in range(nr):
            for col in range(nc):
                idx += 1
                on_tile(idx, total)

                tx = col * STRIDE
                ty = row * STRIDE
                tw = min(STRIDE, w - tx)
                th = min(STRIDE, h - ty)

                cl = min(PAD, tx)
                ct = min(PAD, ty)
                cr = min(PAD, max(0, w - (tx + tw)))
                cb = min(PAD, max(0, h - (ty + th)))

                x1, y1 = tx - cl, ty - ct
                x2, y2 = tx + tw + cr, ty + th + cb
                patch = img[y1:y2, x1:x2]

                pt = PAD - ct
                pl = PAD - cl
                pb = M - patch.shape[0] - pt
                pr = M - patch.shape[1] - pl
                if pt > 0 or pb > 0 or pl > 0 or pr > 0:
                    patch = cv2.copyMakeBorder(patch, pt, pb, pl, pr,
                                               cv2.BORDER_REFLECT_101)

                iex = pl + cl
                iey = pt + ct

                out_tile = self._infer(
                    cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

                ox = iex * 2
                oy = iey * 2
                out[ty * 2:ty * 2 + th * 2,
                    tx * 2:tx * 2 + tw * 2] = out_tile[oy:oy + th * 2,
                                                       ox:ox + tw * 2]
        return out

    def _run(self):
        if self.source_image is None or self.running:
            return
        self.running = True
        self.run_btn.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        t0 = time.time()

        def report(cur, tot):
            el = time.time() - t0
            self.progress.setValue(int(cur / tot * 100))
            self.lbl_tile.setText(f"Tile {cur}/{tot}")
            self.lbl_time.setText(f"{el:.1f}s")
            self.status.setText(f"Processing tile {cur}/{tot} ({el:.1f}s)")
            QApplication.processEvents()

        try:
            h, w = self.source_image.shape[:2]

            self.cmodel.activate()

            if w <= M and h <= M:
                self.status.setText("Running inference…")
                QApplication.processEvents()
                self.result_image = self._run_single(self.source_image)
            else:
                # Show tile count before starting
                nc, nr = tile_grid(w, h)
                self.status.setText(
                    f"Processing {nc * nr} tiles…")
                QApplication.processEvents()
                self.result_image = self._run_tiled(
                    self.source_image, report)

            self.cmodel.deactivate()

            self._show(self.result_image, self.result_lbl)
            el = time.time() - t0
            rh, rw = self.result_image.shape[:2]
            self.status.setText(f"Done — {rw}×{rh} ({el:.1f}s)")
            self.lbl_time.setText(f"{el:.1f}s")
            self.save_btn.setEnabled(True)
        except Exception as e:
            el = time.time() - t0
            try:
                self.cmodel.deactivate()
            except Exception:
                pass
            QMessageBox.critical(self, "Inference Error",
                                 f"Failed after {el:.1f}s:\n{e}")
            self.status.setText(f"Failed ({el:.1f}s)")
        finally:
            self.running = False
            self.run_btn.setEnabled(True)
            self.load_btn.setEnabled(True)
            self.progress.setVisible(False)

    def _save_result(self):
        if self.result_image is None:
            return
        base, _ = os.path.splitext(os.path.basename(self.source_path))
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Result",
            os.path.join(SCRIPT_DIR, f"{base}_x2.png"),
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
        if path:
            cv2.imwrite(path, self.result_image)
            self.status.setText(f"Saved: {os.path.basename(path)}")


def main():
    app = QApplication(sys.argv)
    w = SuperResolutionApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
