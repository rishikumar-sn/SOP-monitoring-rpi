"""Instrumented 120-second Phase 1 acceptance test using the real camera."""

import argparse
from collections import deque
import os
from pathlib import Path
import resource
import statistics
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import CAMERA
from ui.main_window import MainWindow


def current_rss_mib() -> float:
    with open("/proc/self/status", encoding="utf-8") as status_file:
        for line in status_file:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()

    app = QApplication([])
    window = MainWindow(CAMERA)
    window.show()

    started = time.monotonic()
    started_cpu = resource.getrusage(resource.RUSAGE_SELF)
    heartbeats = []
    rss_samples = []
    shapes = set()
    preview_frames = 0
    captured_frames = 0
    measured_capture_fps = deque(maxlen=20)
    camera_info = []
    errors = []
    resize_large = False

    def heartbeat():
        heartbeats.append(time.monotonic())

    def sample():
        nonlocal preview_frames
        frame = window.latest_full_resolution_frame
        if frame is not None:
            shapes.add(frame.shape)
            preview_frames += 1
        rss_samples.append((time.monotonic() - started, current_rss_mib()))

    def record_stats(fps, frames):
        nonlocal captured_frames
        measured_capture_fps.append(fps)
        captured_frames += frames

    def resize_window():
        nonlocal resize_large
        resize_large = not resize_large
        window.resize(1280, 760) if resize_large else window.resize(900, 560)

    def finish():
        window.close()
        app.quit()

    window.camera_worker.camera_opened.connect(camera_info.append)
    window.camera_worker.stats_updated.connect(record_stats)
    window.camera_worker.error.connect(errors.append)
    window.inference_worker.model_failed.connect(errors.append)

    heartbeat_timer = QTimer()
    heartbeat_timer.timeout.connect(heartbeat)
    heartbeat_timer.start(100)

    sample_timer = QTimer()
    sample_timer.timeout.connect(sample)
    sample_timer.start(1000)

    resize_timer = QTimer()
    resize_timer.timeout.connect(resize_window)
    resize_timer.start(5000)

    QTimer.singleShot(args.seconds * 1000, finish)
    app.exec()

    ended = time.monotonic()
    ended_cpu = resource.getrusage(resource.RUSAGE_SELF)
    elapsed = ended - started
    cpu_seconds = (
        ended_cpu.ru_utime
        + ended_cpu.ru_stime
        - started_cpu.ru_utime
        - started_cpu.ru_stime
    )
    cpu_percent = 100.0 * cpu_seconds / elapsed
    max_heartbeat_gap = max(
        (later - earlier for earlier, later in zip(heartbeats, heartbeats[1:])),
        default=float("inf"),
    )

    warm_samples = [rss for seconds, rss in rss_samples if 20 <= seconds <= 40]
    final_samples = [rss for seconds, rss in rss_samples if seconds >= args.seconds - 20]
    memory_drift = (
        statistics.median(final_samples) - statistics.median(warm_samples)
        if warm_samples and final_samples
        else float("inf")
    )
    median_capture_fps = (
        statistics.median(measured_capture_fps)
        if measured_capture_fps
        else 0.0
    )

    print(f"duration_seconds={elapsed:.1f}")
    print(f"camera_info={camera_info[-1] if camera_info else None}")
    print(f"frame_shapes={sorted(shapes)}")
    print(f"capture_frames={captured_frames}")
    print(f"preview_samples={preview_frames}")
    print(f"median_capture_fps={median_capture_fps:.2f}")
    print(f"max_qt_heartbeat_gap_seconds={max_heartbeat_gap:.3f}")
    print(f"cpu_percent_one_core_100={cpu_percent:.1f}")
    print(f"rss_memory_drift_after_warmup_mib={memory_drift:.1f}")
    print(f"errors={errors}")

    failures = []
    if elapsed < args.seconds - 1:
        failures.append("viewer ended before the requested duration")
    if errors:
        failures.append(f"camera errors occurred: {errors}")
    if not camera_info:
        failures.append("camera did not open")
    elif (camera_info[-1].width, camera_info[-1].height) != (2560, 1440):
        failures.append("camera did not negotiate 2560x1440")
    if shapes != {(1440, 2560, 3)}:
        failures.append(f"unexpected full-resolution frame shapes: {shapes}")
    if median_capture_fps < 15.0:
        failures.append(f"capture FPS is too low for live preview: {median_capture_fps:.2f}")
    if max_heartbeat_gap > 1.5:
        failures.append(f"Qt event loop stalled for {max_heartbeat_gap:.3f}s")
    if memory_drift > 32.0:
        failures.append(f"RSS continued growing after warmup: {memory_drift:.1f} MiB")
    if cpu_percent > 150.0:
        failures.append(f"CPU use is unexpectedly high: {cpu_percent:.1f}%")
    if not window._model_ready:
        failures.append("Hailo model did not remain ready")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: Phase 1 camera stability acceptance checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
