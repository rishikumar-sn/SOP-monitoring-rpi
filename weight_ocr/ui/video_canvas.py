import cv2
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget


class VideoCanvas(QWidget):
    roi_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMouseTracking(True)
        self._pixmap = None
        self._message = "Opening camera…"
        self._drawing_enabled = False
        self._roi = None
        self._drag_start = None
        self._drag_current = None

    @property
    def roi(self):
        return self._roi

    def set_bgr_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_frame.shape
        image = QImage(
            rgb_frame.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        )
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_message(self, message: str):
        self._message = message
        self._pixmap = None
        self.update()

    def set_drawing_enabled(self, enabled: bool):
        self._drawing_enabled = enabled
        self.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def clear_roi(self):
        self._roi = None
        self._drag_start = None
        self._drag_current = None
        self.update()

    def set_roi(self, roi, emit=True):
        if self._pixmap is None:
            raise RuntimeError("Cannot set an ROI before a camera frame is displayed")
        x1, y1, x2, y2 = roi
        width = self._pixmap.width()
        height = self._pixmap.height()
        x1 = max(0, min(width, int(x1)))
        x2 = max(0, min(width, int(x2)))
        y1 = max(0, min(height, int(y1)))
        y2 = max(0, min(height, int(y2)))
        normalized = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
            raise ValueError("ROI must have a positive width and height")
        self._roi = normalized
        self.update()
        if emit:
            self.roi_changed.emit(normalized)

    def displayed_image_rect(self):
        if self._pixmap is None or self.width() <= 0 or self.height() <= 0:
            return QRectF()
        scale = min(
            self.width() / self._pixmap.width(),
            self.height() / self._pixmap.height(),
        )
        displayed_width = self._pixmap.width() * scale
        displayed_height = self._pixmap.height() * scale
        return QRectF(
            (self.width() - displayed_width) / 2.0,
            (self.height() - displayed_height) / 2.0,
            displayed_width,
            displayed_height,
        )

    def display_to_source(self, point: QPointF, clamp=False):
        if self._pixmap is None:
            return None
        image_rect = self.displayed_image_rect()
        if not clamp and not image_rect.contains(point):
            return None
        display_x = min(max(point.x(), image_rect.left()), image_rect.right())
        display_y = min(max(point.y(), image_rect.top()), image_rect.bottom())
        source_x = (display_x - image_rect.left()) * (
            self._pixmap.width() / image_rect.width()
        )
        source_y = (display_y - image_rect.top()) * (
            self._pixmap.height() / image_rect.height()
        )
        return QPointF(source_x, source_y)

    def source_to_display(self, point: QPointF):
        if self._pixmap is None:
            return None
        image_rect = self.displayed_image_rect()
        return QPointF(
            image_rect.left()
            + point.x() * image_rect.width() / self._pixmap.width(),
            image_rect.top()
            + point.y() * image_rect.height() / self._pixmap.height(),
        )

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111111"))
        if self._pixmap is None:
            painter.setPen(QColor("#dddddd"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                self._message,
            )
            return

        image_rect = self.displayed_image_rect()
        painter.drawPixmap(image_rect, self._pixmap, QRectF(self._pixmap.rect()))

        roi = self._roi
        if self._drag_start is not None and self._drag_current is not None:
            roi = (
                self._drag_start.x(),
                self._drag_start.y(),
                self._drag_current.x(),
                self._drag_current.y(),
            )
        if roi is not None:
            x1, y1, x2, y2 = roi
            first = self.source_to_display(QPointF(min(x1, x2), min(y1, y2)))
            second = self.source_to_display(QPointF(max(x1, x2), max(y1, y2)))
            painter.setPen(QPen(QColor("#00ff66"), 3))
            painter.drawRect(QRectF(first, second))

    def mousePressEvent(self, event):
        if (
            not self._drawing_enabled
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        source_point = self.display_to_source(event.position())
        if source_point is None:
            return
        self._drag_start = source_point
        self._drag_current = source_point
        self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            return
        self._drag_current = self.display_to_source(event.position(), clamp=True)
        self.update()

    def mouseReleaseEvent(self, event):
        if (
            self._drag_start is None
            or event.button() != Qt.MouseButton.LeftButton
        ):
            return
        self._drag_current = self.display_to_source(event.position(), clamp=True)
        x1 = int(round(min(self._drag_start.x(), self._drag_current.x())))
        y1 = int(round(min(self._drag_start.y(), self._drag_current.y())))
        x2 = int(round(max(self._drag_start.x(), self._drag_current.x())))
        y2 = int(round(max(self._drag_start.y(), self._drag_current.y())))
        self._drag_start = None
        self._drag_current = None
        if x2 > x1 and y2 > y1:
            self.set_roi((x1, y1, x2, y2))
        else:
            self.update()
