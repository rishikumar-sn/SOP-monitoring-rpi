import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from ui.video_canvas import VideoCanvas


class CoordinateMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.canvas = VideoCanvas()
        self.canvas.resize(1280, 800)
        self.canvas.set_bgr_frame(np.zeros((1440, 2560, 3), dtype=np.uint8))
        self.canvas.show()
        self.app.processEvents()

    def tearDown(self):
        self.canvas.close()

    def test_black_bars_are_not_camera_coordinates(self):
        image_rect = self.canvas.displayed_image_rect()
        self.assertAlmostEqual(image_rect.top(), 40.0)
        self.assertAlmostEqual(image_rect.height(), 720.0)
        self.assertIsNone(self.canvas.display_to_source(QPointF(640, 10)))
        self.assertIsNone(self.canvas.display_to_source(QPointF(640, 790)))

    def test_source_display_roundtrip(self):
        for source_point in (
            QPointF(0, 0),
            QPointF(2558, 0),
            QPointF(0, 1438),
            QPointF(2558, 1438),
            QPointF(1280, 720),
        ):
            display_point = self.canvas.source_to_display(source_point)
            roundtrip = self.canvas.display_to_source(display_point)
            self.assertAlmostEqual(roundtrip.x(), source_point.x(), places=6)
            self.assertAlmostEqual(roundtrip.y(), source_point.y(), places=6)

    def test_drawn_rois_map_to_all_five_image_positions(self):
        rois = {
            "top-left": (100, 100, 500, 400),
            "top-right": (2000, 100, 2500, 400),
            "bottom-left": (100, 1000, 500, 1350),
            "bottom-right": (2000, 1000, 2500, 1350),
            "center": (900, 500, 1600, 1000),
        }
        for name, expected_roi in rois.items():
            with self.subTest(position=name):
                self.canvas.clear_roi()
                self.canvas.set_drawing_enabled(True)
                start = self.canvas.source_to_display(
                    QPointF(expected_roi[0], expected_roi[1])
                ).toPoint()
                end = self.canvas.source_to_display(
                    QPointF(expected_roi[2], expected_roi[3])
                ).toPoint()
                QTest.mousePress(
                    self.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=QPoint(start),
                )
                QTest.mouseMove(self.canvas, QPoint(end))
                QTest.mouseRelease(
                    self.canvas,
                    Qt.MouseButton.LeftButton,
                    pos=QPoint(end),
                )
                self.assertEqual(self.canvas.roi, expected_roi)


if __name__ == "__main__":
    unittest.main()
