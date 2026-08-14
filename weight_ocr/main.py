import logging
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from config import CAMERA
from ui.main_window import MainWindow


def configure_logging():
    log_path = Path(__file__).resolve().parent / "logs" / "lcd_weight_reader.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    window = MainWindow(CAMERA)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
