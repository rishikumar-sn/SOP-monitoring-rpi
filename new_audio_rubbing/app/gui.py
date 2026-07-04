from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFileSystemModel, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.io import wavfile

from app.audio_recorder import device_details, list_input_devices, record_audio
from app.dataset_manager import DatasetManager
from app.predictor import AudioPredictor
from app.trainer import TrainingConfig, train_model
from app.feature_extractor import FeatureConfig
from app.utils import (
    CLASSES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DURATION,
    DEFAULT_SAMPLE_RATE,
    dataset_root,
    open_in_file_manager,
)


class RecordingWorker(QObject):
    completed = pyqtSignal(object)
    audio_chunk = pyqtSignal(np.ndarray)  # Emit audio chunks during recording
    failed = pyqtSignal(str)

    def __init__(self, device_index: int, duration: float, sample_rate: int) -> None:
        super().__init__()
        self.device_index = device_index
        self.duration = duration
        self.sample_rate = sample_rate

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.completed.emit(
                record_audio(self.device_index, self.duration, self.sample_rate, self.audio_chunk.emit)
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class TrainingWorker(QObject):
    log_message = pyqtSignal(str)
    progress_changed = pyqtSignal(int)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, manager: DatasetManager, config: TrainingConfig) -> None:
        super().__init__()
        self.manager = manager
        self.config = config

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = train_model(
                self.manager,
                self.config,
                self.log_message.emit,
                self.progress_changed.emit,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class PredictionWorker(QObject):
    prediction_ready = pyqtSignal(str, float, object)
    stream_started = pyqtSignal(object)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, device_index: int, interval_sec: float) -> None:
        super().__init__()
        self.device_index = device_index
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @pyqtSlot()
    def run(self) -> None:
        try:
            predictor = AudioPredictor()
            predictor.load()
            duration = float(predictor.config["duration"])
            sample_rate = int(predictor.config["sample_rate"])
            window_samples = max(1, int(round(duration * sample_rate)))
            step_samples = max(1, int(round(self.interval_sec * sample_rate)))
            block_samples = min(step_samples, max(256, sample_rate // 20))
            rolling_audio = np.empty(0, dtype=np.float32)
            samples_since_prediction = 0
            device_info = sd.query_devices(self.device_index, "input")
            device_default_rate = float(device_info.get("default_samplerate", 0.0))

            with sd.InputStream(
                device=self.device_index,
                channels=1,
                samplerate=sample_rate,
                blocksize=block_samples,
                dtype="float32",
            ) as stream:
                actual_stream_rate = float(stream.samplerate)
                self.stream_started.emit(
                    {
                        "duration": duration,
                        "device_name": str(device_info.get("name", self.device_index)),
                        "device_default_rate": device_default_rate,
                        "requested_rate": sample_rate,
                        "actual_stream_rate": actual_stream_rate,
                        "model_rate": sample_rate,
                        "python_resampling": False,
                        "host_api": str(
                            sd.query_hostapis(int(device_info.get("hostapi", 0))).get(
                                "name",
                                "",
                            )
                        ),
                    }
                )
                while not self._stop_event.is_set():
                    chunk, overflowed = stream.read(block_samples)
                    if overflowed:
                        continue
                    mono = np.nan_to_num(
                        np.asarray(chunk[:, 0], dtype=np.float32),
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0,
                    )
                    rolling_audio = np.concatenate((rolling_audio, mono))
                    if rolling_audio.size > window_samples:
                        rolling_audio = rolling_audio[-window_samples:]
                    samples_since_prediction += mono.size

                    if (
                        rolling_audio.size == window_samples
                        and samples_since_prediction >= step_samples
                    ):
                        predicted, confidence, probabilities = predictor.predict(
                            rolling_audio.copy()
                        )
                        self.prediction_ready.emit(
                            predicted,
                            confidence,
                            probabilities,
                        )
                        samples_since_prediction = 0
        except Exception as exc:
            if not self._stop_event.is_set():
                self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class WaveformWidget(FigureCanvasQTAgg):
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.figure = Figure(figsize=(10, 3), dpi=100, facecolor='#101820', edgecolor='#101820')
        super().__init__(self.figure)
        self.setMinimumHeight(250)
        self.audio = np.zeros(1, dtype=np.float32)
        self.sample_rate = sample_rate
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#101820')
        self.ax.set_xlabel('Time (s)', color='#ffffff')
        self.ax.set_ylabel('Amplitude', color='#ffffff')
        self.ax.tick_params(colors='#ffffff')
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    def set_audio(self, audio: np.ndarray) -> None:
        self.audio = np.asarray(audio, dtype=np.float32)
        self._update_plot()

    def add_audio_chunk(self, chunk: np.ndarray) -> None:
        """Append audio chunk and update plot for live visualization."""
        chunk = np.asarray(chunk, dtype=np.float32)
        self.audio = np.concatenate([self.audio, chunk])
        self._update_plot()

    def _update_plot(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor('#101820')
        if self.audio.size > 0:
            time = np.arange(self.audio.size) / self.sample_rate
            self.ax.plot(time, self.audio, color='#2dd4bf', linewidth=0.8)
            self.ax.set_xlabel('Time (s)', color='#ffffff')
            self.ax.set_ylabel('Amplitude', color='#ffffff')
            self.ax.tick_params(colors='#ffffff')
            self.ax.grid(True, alpha=0.2, color='#2dd4bf')
        self.draw()

    def reset(self) -> None:
        """Reset waveform for new recording."""
        self.audio = np.zeros(1, dtype=np.float32)
        self._update_plot()


def populate_microphones(combo: QComboBox, info_label: QLabel | None = None) -> None:
    previous = combo.currentData()
    combo.clear()
    try:
        devices = list_input_devices()
        for device in devices:
            combo.addItem(device.display_name, device.index)
        if previous is not None:
            index = combo.findData(previous)
            if index >= 0:
                combo.setCurrentIndex(index)
        if not devices and info_label is not None:
            info_label.setText("No input microphones were found.")
    except Exception as exc:
        if info_label is not None:
            info_label.setText(f"Could not query microphones: {exc}")


class CollectionTab(QWidget):
    counts_changed = pyqtSignal()

    def __init__(self, manager: DatasetManager) -> None:
        super().__init__()
        self.manager = manager
        self.record_thread: QThread | None = None
        self.record_worker: RecordingWorker | None = None
        self.auto_capture = False
        self.pending_capture: dict[str, object] = {}
        self._build_ui()
        self.refresh_microphones()
        self.refresh_counts()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        microphone_group = QGroupBox("Microphone")
        microphone_layout = QGridLayout(microphone_group)
        self.mic_combo = QComboBox()
        refresh_mic = QPushButton("Refresh")
        refresh_mic.clicked.connect(self.refresh_microphones)
        self.mic_info = QLabel()
        self.mic_info.setWordWrap(True)
        self.mic_combo.currentIndexChanged.connect(self.update_mic_info)
        microphone_layout.addWidget(self.mic_combo, 0, 0)
        microphone_layout.addWidget(refresh_mic, 0, 1)
        microphone_layout.addWidget(self.mic_info, 1, 0, 1, 2)
        layout.addWidget(microphone_group)

        center = QSplitter(Qt.Orientation.Horizontal)
        class_group = QGroupBox("Sample Class")
        class_layout = QVBoxLayout(class_group)
        self.class_list = QListWidget()
        self.class_list.addItems(CLASSES)
        self.class_list.setCurrentRow(0)
        self.class_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        class_layout.addWidget(self.class_list)
        self.counts_label = QLabel()
        self.counts_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        class_layout.addWidget(self.counts_label)
        center.addWidget(class_group)

        settings_group = QGroupBox("Capture Settings")
        form = QFormLayout(settings_group)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 30.0)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setValue(DEFAULT_DURATION)
        self.duration_spin.setSuffix(" sec")
        self.delay_spin = QDoubleSpinBox()
        self.delay_spin.setRange(0.0, 60.0)
        self.delay_spin.setDecimals(1)
        self.delay_spin.setValue(1.0)
        self.delay_spin.setSuffix(" sec")
        self.operator_edit = QLineEdit()
        self.operator_edit.setPlaceholderText("operatorA")
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(90)
        form.addRow("Duration:", self.duration_spin)
        form.addRow("Auto-capture delay:", self.delay_spin)
        form.addRow("Operator/session:", self.operator_edit)
        form.addRow("Notes:", self.notes_edit)
        center.addWidget(settings_group)
        layout.addWidget(center)

        buttons = QHBoxLayout()
        self.record_button = QPushButton("Record One Sample")
        self.auto_start_button = QPushButton("Start Auto Capture")
        self.auto_stop_button = QPushButton("Stop Auto Capture")
        self.auto_stop_button.setEnabled(False)
        self.record_button.clicked.connect(lambda: self.start_recording(False))
        self.auto_start_button.clicked.connect(self.start_auto_capture)
        self.auto_stop_button.clicked.connect(self.stop_auto_capture)
        create_button = QPushButton("Create Dataset Folders")
        create_button.clicked.connect(self.create_folders)
        open_button = QPushButton("Open Dataset Folder")
        open_button.clicked.connect(lambda: open_in_file_manager(self.manager.root))
        counts_button = QPushButton("Refresh Counts")
        counts_button.clicked.connect(self.refresh_counts)
        for button in (
            self.record_button,
            self.auto_start_button,
            self.auto_stop_button,
            create_button,
            open_button,
            counts_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.status_label = QLabel("Ready.")
        self.status_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Waveform Preview"))
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

    def create_folders(self) -> None:
        try:
            self.manager.create_folders()
            self.refresh_counts()
            self.status_label.setText(f"Dataset folders ready: {self.manager.root}")
        except Exception as exc:
            QMessageBox.critical(self, "Dataset Error", str(exc))

    def refresh_microphones(self) -> None:
        populate_microphones(self.mic_combo, self.mic_info)
        self.update_mic_info()

    def update_mic_info(self) -> None:
        device_index = self.mic_combo.currentData()
        if device_index is None:
            return
        try:
            info = device_details(int(device_index))
            self.mic_info.setText(
                f"{info.get('name')} | Input channels: "
                f"{info.get('max_input_channels')} | Default rate: "
                f"{info.get('default_samplerate')} Hz"
            )
        except Exception as exc:
            self.mic_info.setText(f"Device information unavailable: {exc}")

    def refresh_counts(self) -> None:
        counts = self.manager.counts()
        self.counts_label.setText(
            "\n".join(f"{name}: {counts[name]}" for name in CLASSES)
        )
        self.counts_changed.emit()

    def start_auto_capture(self) -> None:
        if self.record_thread is not None:
            return
        self.auto_capture = True
        self.auto_start_button.setEnabled(False)
        self.auto_stop_button.setEnabled(True)
        self.start_recording(True)

    def stop_auto_capture(self) -> None:
        self.auto_capture = False
        self.auto_start_button.setEnabled(True)
        self.auto_stop_button.setEnabled(False)
        if self.record_thread is not None:
            self.status_label.setText("Stopping after the current recording...")
        else:
            self.status_label.setText("Auto capture stopped.")

    def start_recording(self, auto: bool = False) -> None:
        if self.record_thread is not None:
            return
        device_index = self.mic_combo.currentData()
        selected_class = self.class_list.currentItem()
        if device_index is None:
            QMessageBox.warning(self, "No Microphone", "Select an input microphone.")
            self.stop_auto_capture()
            return
        if selected_class is None:
            QMessageBox.warning(self, "No Class", "Select a sample class.")
            self.stop_auto_capture()
            return

        self.record_button.setEnabled(False)
        self.status_label.setText(
            f"Recording {selected_class.text()} for {self.duration_spin.value():.2f} seconds..."
        )
        self.waveform.reset()  # Reset waveform for new recording
        self.pending_capture = {
            "class_name": selected_class.text(),
            "operator": self.operator_edit.text(),
            "duration": self.duration_spin.value(),
            "mic_name": self.mic_combo.currentText().split(": ", 1)[-1],
            "notes": self.notes_edit.toPlainText(),
        }
        thread = QThread(self)
        worker = RecordingWorker(
            int(device_index),
            self.duration_spin.value(),
            DEFAULT_SAMPLE_RATE,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.audio_chunk.connect(self.waveform.add_audio_chunk)  # Live waveform updates
        worker.completed.connect(self.recording_completed)
        worker.failed.connect(self.recording_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self.recording_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.record_thread = thread
        self.record_worker = worker
        thread.start()

    @pyqtSlot(object)
    def recording_completed(self, audio: np.ndarray) -> None:
        class_name = str(self.pending_capture["class_name"])
        try:
            path, metadata = self.manager.save_sample(
                audio=audio,
                class_name=class_name,
                operator=str(self.pending_capture["operator"]),
                duration_sec=float(self.pending_capture["duration"]),
                sample_rate=DEFAULT_SAMPLE_RATE,
                mic_device_name=str(self.pending_capture["mic_name"]),
                notes=str(self.pending_capture["notes"]),
            )
            self.waveform.set_audio(audio)
            rms = float(metadata["rms_energy"])
            peak = float(metadata["peak_amplitude"])
            warnings = []
            if peak >= 0.98:
                warnings.append("Warning: signal is clipping or very close to clipping.")
            if rms < 0.003 and class_name != "NO_RUB_NOK":
                warnings.append("Warning: RMS energy is very low.")
            suffix = " " + " ".join(warnings) if warnings else ""
            self.status_label.setText(f"Saved {path.name}.{suffix}")
            self.refresh_counts()
        except Exception as exc:
            self.recording_failed(str(exc))

    @pyqtSlot(str)
    def recording_failed(self, message: str) -> None:
        self.status_label.setText(f"Recording failed: {message}")
        self.auto_capture = False
        self.auto_start_button.setEnabled(True)
        self.auto_stop_button.setEnabled(False)
        QMessageBox.critical(self, "Recording Error", message)

    def recording_thread_finished(self) -> None:
        self.record_thread = None
        self.record_worker = None
        self.record_button.setEnabled(True)
        if self.auto_capture:
            delay_ms = int(self.delay_spin.value() * 1000)
            self.status_label.setText(f"Waiting {self.delay_spin.value():.1f} seconds...")
            QTimer.singleShot(delay_ms, self._continue_auto_capture)

    def _continue_auto_capture(self) -> None:
        if self.auto_capture and self.record_thread is None:
            self.start_recording(True)


class BrowserTab(QWidget):
    def __init__(self, manager: DatasetManager) -> None:
        super().__init__()
        self.manager = manager
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh)
        play_button = QPushButton("Play Selected Sample")
        play_button.clicked.connect(self.play_selected)
        delete_button = QPushButton("Delete Selected Sample")
        delete_button.clicked.connect(self.delete_selected)
        open_button = QPushButton("Open Dataset Folder")
        open_button.clicked.connect(lambda: open_in_file_manager(self.manager.root))
        for button in (refresh_button, play_button, delete_button, open_button):
            controls.addWidget(button)
        layout.addLayout(controls)

        self.counts_label = QLabel()
        layout.addWidget(self.counts_label)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.file_model = QFileSystemModel(self)
        self.file_model.setNameFilters(["*.wav"])
        self.file_model.setNameFilterDisables(False)
        self.tree = QTreeView()
        self.tree.setModel(self.file_model)
        self.tree.clicked.connect(self.selection_changed)
        self.tree.doubleClicked.connect(self.play_selected)
        splitter.addWidget(self.tree)
        metadata_group = QGroupBox("Selected Sample Metadata")
        metadata_layout = QVBoxLayout(metadata_group)
        self.metadata_text = QTextEdit()
        self.metadata_text.setReadOnly(True)
        metadata_layout.addWidget(self.metadata_text)
        splitter.addWidget(metadata_group)
        splitter.setSizes([650, 350])
        layout.addWidget(splitter)

    def refresh(self) -> None:
        self.manager.create_folders()
        root_index = self.file_model.setRootPath(str(self.manager.root))
        self.tree.setRootIndex(root_index)
        self.counts_label.setText(
            " | ".join(
                f"{name}: {count}" for name, count in self.manager.counts().items()
            )
        )

    def selected_path(self) -> Path | None:
        indexes = self.tree.selectionModel().selectedRows()
        if not indexes:
            return None
        return Path(self.file_model.filePath(indexes[0]))

    def selection_changed(self) -> None:
        path = self.selected_path()
        if path is None or not path.is_file():
            self.metadata_text.clear()
            return
        metadata = self.manager.metadata_for(path)
        if metadata:
            self.metadata_text.setPlainText(
                "\n".join(f"{key}: {value}" for key, value in metadata.items())
            )
        else:
            self.metadata_text.setPlainText(
                f"filepath: {path}\nNo metadata row was found."
            )

    def play_selected(self) -> None:
        path = self.selected_path()
        if path is None or path.suffix.lower() != ".wav":
            QMessageBox.information(self, "Select Sample", "Select a WAV sample first.")
            return
        try:
            sample_rate, audio = wavfile.read(path)
            if np.issubdtype(audio.dtype, np.integer):
                scale = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
                audio = audio.astype(np.float32) / scale
            sd.stop()
            sd.play(audio, sample_rate, blocking=False)
        except Exception as exc:
            QMessageBox.critical(self, "Playback Error", str(exc))

    def delete_selected(self) -> None:
        path = self.selected_path()
        if path is None or path.suffix.lower() != ".wav":
            QMessageBox.information(self, "Select Sample", "Select a WAV sample first.")
            return
        answer = QMessageBox.question(
            self,
            "Delete Sample",
            f"Delete {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            try:
                self.manager.delete_sample(path)
                self.metadata_text.clear()
                self.refresh()
            except Exception as exc:
                QMessageBox.critical(self, "Delete Error", str(exc))


class TrainingTab(QWidget):
    def __init__(self, manager: DatasetManager) -> None:
        super().__init__()
        self.manager = manager
        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self._build_ui()
        self.scan_dataset()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        scan_button = QPushButton("Scan Dataset")
        scan_button.clicked.connect(self.scan_dataset)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        top.addWidget(scan_button)
        top.addWidget(self.start_button)
        top.addStretch()
        layout.addLayout(top)

        self.distribution_label = QLabel()
        self.distribution_label.setWordWrap(True)
        layout.addWidget(self.distribution_label)
        parameters = QGroupBox("Training Parameters")
        form = QFormLayout(parameters)
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 96000)
        self.sample_rate_spin.setValue(DEFAULT_SAMPLE_RATE)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10.0)
        self.duration_spin.setValue(DEFAULT_DURATION)
        self.n_mels_spin = QSpinBox()
        self.n_mels_spin.setRange(16, 256)
        self.n_mels_spin.setValue(64)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(32)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(50)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 200)
        self.patience_spin.setValue(15)
        self.minimum_epochs_spin = QSpinBox()
        self.minimum_epochs_spin.setRange(1, 1000)
        self.minimum_epochs_spin.setValue(15)
        self.validation_spin = QDoubleSpinBox()
        self.validation_spin.setRange(0.05, 0.5)
        self.validation_spin.setSingleStep(0.05)
        self.validation_spin.setValue(0.2)
        form.addRow("Sample rate:", self.sample_rate_spin)
        form.addRow("Duration (sec):", self.duration_spin)
        form.addRow("Mel bins:", self.n_mels_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow("Maximum epochs:", self.epochs_spin)
        form.addRow("Minimum epochs:", self.minimum_epochs_spin)
        form.addRow("Early-stop patience:", self.patience_spin)
        form.addRow("Validation split:", self.validation_spin)
        layout.addWidget(parameters)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        bottom.addWidget(self.log)
        self.figure = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        bottom.addWidget(self.canvas)
        bottom.setSizes([450, 650])
        layout.addWidget(bottom)

    def scan_dataset(self) -> None:
        counts = self.manager.counts()
        total = sum(counts.values())
        self.distribution_label.setText(
            f"Total samples: {total}\n"
            + " | ".join(f"{name}: {count}" for name, count in counts.items())
        )

    def start_training(self) -> None:
        if self.training_thread is not None:
            return
        counts = self.manager.counts()
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            QMessageBox.warning(
                self,
                "Incomplete Dataset",
                "Add samples for every class before training.\nMissing: "
                + ", ".join(missing),
            )
            return

        feature_config = FeatureConfig(
            sample_rate=self.sample_rate_spin.value(),
            duration=self.duration_spin.value(),
            n_mels=self.n_mels_spin.value(),
            n_fft=512,
            hop_length=160,
            win_length=400,
        )
        config = TrainingConfig(
            features=feature_config,
            batch_size=self.batch_spin.value(),
            epochs=self.epochs_spin.value(),
            validation_split=self.validation_spin.value(),
            early_stopping_patience=self.patience_spin.value(),
            minimum_epochs=min(
                self.minimum_epochs_spin.value(),
                self.epochs_spin.value(),
            ),
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        )
        self.log.clear()
        self.progress.setValue(0)
        self.start_button.setEnabled(False)
        self.log.append("Starting training...")
        thread = QThread(self)
        worker = TrainingWorker(self.manager, config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log_message.connect(self.log.append)
        worker.progress_changed.connect(self.progress.setValue)
        worker.completed.connect(self.training_completed)
        worker.failed.connect(self.training_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self.training_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.training_thread = thread
        self.training_worker = worker
        thread.start()

    @pyqtSlot(object)
    def training_completed(self, result: dict[str, object]) -> None:
        self.log.append("Training and model export completed.")
        self.plot_results(result)

    @pyqtSlot(str)
    def training_failed(self, message: str) -> None:
        self.log.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Training Error", message)

    def training_thread_finished(self) -> None:
        self.training_thread = None
        self.training_worker = None
        self.start_button.setEnabled(True)
        self.scan_dataset()

    def plot_results(self, result: dict[str, object]) -> None:
        history = result["history"]
        matrix = np.asarray(result["confusion_matrix"])
        self.figure.clear()
        accuracy_axis = self.figure.add_subplot(2, 2, 1)
        loss_axis = self.figure.add_subplot(2, 2, 2)
        matrix_axis = self.figure.add_subplot(2, 1, 2)
        accuracy_axis.plot(history.get("accuracy", []), label="train")
        accuracy_axis.plot(history.get("val_accuracy", []), label="validation")
        accuracy_axis.set_title("Accuracy")
        accuracy_axis.legend()
        loss_axis.plot(history.get("loss", []), label="train")
        loss_axis.plot(history.get("val_loss", []), label="validation")
        loss_axis.set_title("Loss")
        loss_axis.legend()
        image = matrix_axis.imshow(matrix, cmap="Blues")
        matrix_axis.set_title("Validation Confusion Matrix")
        matrix_axis.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
        matrix_axis.set_yticks(range(len(CLASSES)), labels=CLASSES)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                matrix_axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
        self.figure.colorbar(image, ax=matrix_axis, fraction=0.025)
        self.figure.tight_layout()
        self.canvas.draw_idle()


class PredictionTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.prediction_thread: QThread | None = None
        self.prediction_worker: PredictionWorker | None = None
        self._build_ui()
        self.refresh_microphones()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        mic_row = QHBoxLayout()
        self.mic_combo = QComboBox()
        refresh_button = QPushButton("Refresh Microphones")
        refresh_button.clicked.connect(self.refresh_microphones)
        mic_row.addWidget(QLabel("Microphone:"))
        mic_row.addWidget(self.mic_combo, 1)
        mic_row.addWidget(refresh_button)
        layout.addLayout(mic_row)

        threshold_row = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(DEFAULT_CONFIDENCE_THRESHOLD)
        threshold_row.addWidget(QLabel("OK confidence threshold:"))
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addStretch()
        layout.addLayout(threshold_row)

        interval_row = QHBoxLayout()
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.10, 5.0)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setSingleStep(0.10)
        self.interval_spin.setValue(0.25)
        self.interval_spin.setSuffix(" sec")
        interval_row.addWidget(QLabel("Prediction update interval:"))
        interval_row.addWidget(self.interval_spin)
        interval_row.addStretch()
        layout.addLayout(interval_row)

        prediction_buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Real-Time Prediction")
        self.start_button.setMinimumHeight(45)
        self.start_button.clicked.connect(self.start_prediction)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(45)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_prediction)
        prediction_buttons.addWidget(self.start_button)
        prediction_buttons.addWidget(self.stop_button)
        layout.addLayout(prediction_buttons)
        self.status_label = QLabel("Ready.")
        layout.addWidget(self.status_label)

        diagnostics = QGroupBox("Audio Rate Diagnostics")
        diagnostics_form = QFormLayout(diagnostics)
        self.device_rate_label = QLabel("-")
        self.stream_rate_label = QLabel("-")
        self.model_rate_label = QLabel("-")
        self.resampling_label = QLabel("-")
        self.audio_route_label = QLabel("-")
        self.audio_route_label.setWordWrap(True)
        diagnostics_form.addRow("Device default/native rate:", self.device_rate_label)
        diagnostics_form.addRow("Actual opened stream rate:", self.stream_rate_label)
        diagnostics_form.addRow("Model-required rate:", self.model_rate_label)
        diagnostics_form.addRow("Python resampling:", self.resampling_label)
        diagnostics_form.addRow("Device / host API:", self.audio_route_label)
        layout.addWidget(diagnostics)

        results = QGroupBox("Prediction")
        result_layout = QFormLayout(results)
        self.class_label = QLabel("-")
        self.confidence_label = QLabel("-")
        self.decision_label = QLabel("-")
        self.decision_label.setStyleSheet("font-size: 30px; font-weight: bold;")
        result_layout.addRow("Predicted class:", self.class_label)
        result_layout.addRow("Confidence:", self.confidence_label)
        result_layout.addRow("Final result:", self.decision_label)
        layout.addWidget(results)

        self.probabilities = QTextEdit()
        self.probabilities.setReadOnly(True)
        layout.addWidget(QLabel("All Class Probabilities"))
        layout.addWidget(self.probabilities)
        warning = QLabel(
            "Do not use only softmax confidence for final production. "
            "Add embedding-distance/anomaly rejection later."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #b45309; font-weight: bold;")
        layout.addWidget(warning)

    def refresh_microphones(self) -> None:
        populate_microphones(self.mic_combo)

    def start_prediction(self) -> None:
        if self.prediction_thread is not None:
            return
        device_index = self.mic_combo.currentData()
        if device_index is None:
            QMessageBox.warning(self, "No Microphone", "Select an input microphone.")
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.mic_combo.setEnabled(False)
        self.interval_spin.setEnabled(False)
        self.status_label.setText("Loading model and opening microphone...")
        thread = QThread(self)
        worker = PredictionWorker(
            int(device_index),
            self.interval_spin.value(),
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.stream_started.connect(self.stream_started)
        worker.prediction_ready.connect(self.prediction_completed)
        worker.failed.connect(self.prediction_failed)
        worker.finished.connect(thread.quit)
        thread.finished.connect(self.prediction_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self.prediction_thread = thread
        self.prediction_worker = worker
        thread.start()

    @pyqtSlot(object)
    def stream_started(self, diagnostics: dict[str, object]) -> None:
        duration = float(diagnostics["duration"])
        device_rate = float(diagnostics["device_default_rate"])
        actual_rate = float(diagnostics["actual_stream_rate"])
        model_rate = float(diagnostics["model_rate"])
        requested_rate = float(diagnostics["requested_rate"])
        python_resampling = bool(diagnostics["python_resampling"])

        self.device_rate_label.setText(f"{device_rate:.0f} Hz")
        self.stream_rate_label.setText(
            f"{actual_rate:.0f} Hz (requested {requested_rate:.0f} Hz)"
        )
        self.model_rate_label.setText(f"{model_rate:.0f} Hz")
        self.resampling_label.setText("YES" if python_resampling else "NO")
        self.audio_route_label.setText(
            f"{diagnostics['device_name']} | {diagnostics['host_api']}"
        )

        conversion_note = (
            "No Python conversion"
            if actual_rate == model_rate and not python_resampling
            else "Rate mismatch detected"
        )
        self.status_label.setText(
            f"Real-time prediction active: {duration:.2f}-second rolling "
            f"window; stream={actual_rate:.0f} Hz, model={model_rate:.0f} Hz. "
            f"{conversion_note}."
        )
    def stop_prediction(self) -> None:
        if self.prediction_worker is None:
            return
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopping real-time prediction...")
        self.prediction_worker.stop()

    @pyqtSlot(str, float, object)
    def prediction_completed(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict[str, float],
    ) -> None:
        is_ok = (
            predicted_class == "JEWEL_RUB_OK"
            and confidence >= self.threshold_spin.value()
        )
        decision = "OK" if is_ok else "NOK"
        color = "#15803d" if is_ok else "#b91c1c"
        self.class_label.setText(predicted_class)
        self.confidence_label.setText(f"{confidence:.2%}")
        self.decision_label.setText(decision)
        self.decision_label.setStyleSheet(
            f"font-size: 30px; font-weight: bold; color: {color};"
        )
        sorted_values = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        self.probabilities.setPlainText(
            "\n".join(f"{name}: {probability:.4%}" for name, probability in sorted_values)
        )
        self.status_label.setText("Real-time prediction active.")

    @pyqtSlot(str)
    def prediction_failed(self, message: str) -> None:
        self.status_label.setText(f"Prediction failed: {message}")
        QMessageBox.critical(self, "Prediction Error", message)

    def prediction_thread_finished(self) -> None:
        self.prediction_thread = None
        self.prediction_worker = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mic_combo.setEnabled(True)
        self.interval_spin.setEnabled(True)
        if not self.status_label.text().startswith("Prediction failed"):
            self.status_label.setText("Real-time prediction stopped.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gold Rubbing Audio Classifier")
        self.resize(1200, 850)
        self.manager = DatasetManager(dataset_root())
        self.manager.create_folders()
        self.tabs = QTabWidget()
        self.collection_tab = CollectionTab(self.manager)
        self.browser_tab = BrowserTab(self.manager)
        self.training_tab = TrainingTab(self.manager)
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.collection_tab, "Dataset Collection")
        self.tabs.addTab(self.browser_tab, "Dataset Browser")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.prediction_tab, "Real-Time Prediction")
        self.collection_tab.counts_changed.connect(self.browser_tab.refresh)
        self.collection_tab.counts_changed.connect(self.training_tab.scan_dataset)
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage(f"Dataset: {self.manager.root}")

    def closeEvent(self, event) -> None:
        if self.prediction_tab.prediction_worker is not None:
            self.prediction_tab.stop_prediction()
            if self.prediction_tab.prediction_thread is not None:
                self.prediction_tab.prediction_thread.quit()
                self.prediction_tab.prediction_thread.wait(3000)
        active_threads = [
            self.collection_tab.record_thread,
            self.training_tab.training_thread,
        ]
        if any(thread is not None and thread.isRunning() for thread in active_threads):
            QMessageBox.warning(
                self,
                "Operation Running",
                "Stop or wait for recording/training/prediction before closing.",
            )
            event.ignore()
            return
        sd.stop()
        event.accept()


def run_application() -> int:
    application = QApplication.instance() or QApplication([])
    application.setApplicationName("Gold Rubbing Audio Classifier")
    window = MainWindow()
    window.show()
    return application.exec()
