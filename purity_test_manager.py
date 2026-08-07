from __future__ import annotations

import importlib.util
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from hailo_model_runner import (
    DEFAULT_HAILO_BATCH_SIZE,
    HailoRuntime,
)

logger = logging.getLogger(__name__)


FrameGetter = Callable[[], np.ndarray | None]
StatusSetter = Callable[[str], None]
SessionCallback = Callable[[], None]
MAX_CONSECUTIVE_HAILO_ERRORS = 2
try:
    PURITY_HAILO_INFERENCE_TIMEOUT_MS = max(
        10000,
        int(os.environ.get("PURITY_HAILO_INFERENCE_TIMEOUT_MS", "15000")),
    )
except (TypeError, ValueError):
    PURITY_HAILO_INFERENCE_TIMEOUT_MS = 15000
try:
    PURITY_VIDEO_WRITE_FPS = max(
        1.0,
        float(os.environ.get("PURITY_VIDEO_WRITE_FPS", "6")),
    )
except (TypeError, ValueError):
    PURITY_VIDEO_WRITE_FPS = 6.0


@dataclass
class PuritySessionSummary:
    available: bool = True
    models_loaded: bool = False
    running: bool = False
    started_at: str = ""
    stopped_at: str = ""
    completed_at: str = ""
    stage: str = "IDLE"
    result: str = "Not started"
    status: str = "Purity test idle"
    error: str = ""
    audio_mode: str = "sync"
    audio_selected_device: str = "__AUTO__"
    audio_selected_device_value: str | int | None = "__AUTO__"
    audio_device_name: str = ""
    sound_status: str = "Waiting..."
    audio_prediction: str = "Waiting..."
    audio_label: str = "Waiting..."
    audio_decision: str = "Waiting..."
    audio_confidence: float = 0.0
    audio_ok_threshold: float = 0.70
    audio_probabilities: dict[str, float] = field(default_factory=dict)
    audio_input_rate: int = 0
    audio_model_rate: int = 0
    audio_model_backend: str = ""
    audio_debug: str = ""
    inference_status: str = "Idle"
    inference_count: int = 0
    last_inference_stage: str = ""
    last_inference_ms: float = 0.0
    last_inference_at: str = ""
    rubbing_ok: bool = False
    acid_ok: bool = False
    video_path: str = ""
    rubbing_image_path: str = ""
    rubbing_zoom_image_path: str = ""
    acid_stage_image_path: str = ""
    acid_success_image_path: str = ""
    acid_zoom_image_path: str = ""
    final_image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "models_loaded": bool(self.models_loaded),
            "running": bool(self.running),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "completed_at": self.completed_at,
            "stage": self.stage,
            "result": self.result,
            "status": self.status,
            "error": self.error,
            "audio_mode": self.audio_mode,
            "audio_selected_device": self.audio_selected_device,
            "audio_selected_device_value": self.audio_selected_device_value,
            "audio_device_name": self.audio_device_name,
            "sound_status": self.sound_status,
            "audio_prediction": self.audio_prediction,
            "audio_label": self.audio_label,
            "audio_decision": self.audio_decision,
            "audio_confidence": float(self.audio_confidence),
            "audio_ok_threshold": float(self.audio_ok_threshold),
            "audio_probabilities": dict(self.audio_probabilities or {}),
            "audio_input_rate": int(self.audio_input_rate or 0),
            "audio_model_rate": int(self.audio_model_rate or 0),
            "audio_model_backend": self.audio_model_backend,
            "audio_debug": self.audio_debug,
            "inference_status": self.inference_status,
            "inference_count": int(self.inference_count),
            "last_inference_stage": self.last_inference_stage,
            "last_inference_ms": float(self.last_inference_ms),
            "last_inference_at": self.last_inference_at,
            "rubbing_ok": bool(self.rubbing_ok),
            "acid_ok": bool(self.acid_ok),
            "video_path": self.video_path,
            "rubbing_image_path": self.rubbing_image_path,
            "rubbing_zoom_image_path": self.rubbing_zoom_image_path,
            "acid_stage_image_path": self.acid_stage_image_path,
            "acid_success_image_path": self.acid_success_image_path,
            "acid_zoom_image_path": self.acid_zoom_image_path,
            "final_image_path": self.final_image_path,
        }


class PurityHailoAdapter:
    """Add the predict() API from run-local4 to a shared runtime model.

    run-local4.py's preprocessing checks ``self.input_dtype`` to decide
    whether to normalize pixels to [0,1]:
        - input_dtype == float32  →  pixels / 255  →  [0, 1]
        - input_dtype == uint8    →  raw pixels     →  [0, 255]

    The adapter follows the host input format selected by the shared runner.
    """

    def __init__(self, base_model: Any, run_local_module: Any):
        self._base = base_model
        self._module = run_local_module
        self._logged_output_schema = False
        self._logged_decoder = False
        self._warned_no_decode = False
        self._empty_predict_counter = 0
        # Keep preprocessing aligned with the host format selected by the
        # shared runner. FLOAT32 uses normalized pixels; UINT8 uses raw pixels.
        self.input_dtype = self._base.input_dtype

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def _run_inference(self, input_tensor: np.ndarray) -> Any:
        return self._base.run_inference(
            np.ascontiguousarray(input_tensor),
            cancel_event=getattr(self, "_cancel_event", None),
        )

    def _prepare_input(self, frame_bgr: np.ndarray):
        return self._module.HailoHEFModel._prepare_input(self, frame_bgr)

    def validate_input_contract(self) -> dict[str, Any]:
        """Validate preprocessing against the HEF without running inference."""
        probe = np.zeros(
            (max(8, int(self.input_h)), max(8, int(self.input_w)), 3),
            dtype=np.uint8,
        )
        prepared, _meta = self._prepare_input(probe)
        bound = self._base._prepare_input_buffer(prepared)
        expected_shape = tuple(int(value) for value in self._base.input_shape)
        expected_bytes = int(np.prod(expected_shape)) * np.dtype(self._base.input_dtype).itemsize
        return {
            "name": self._base.name,
            "layout": self._base.input_layout,
            "shape": tuple(bound.shape),
            "dtype": str(bound.dtype),
            "bytes": int(bound.nbytes),
            "expected_shape": expected_shape,
            "expected_bytes": expected_bytes,
        }

    def validate_runtime_contract(self, probe_runs: int = 1) -> dict[str, Any]:
        """Verify that repeated Hailo submissions complete on every output channel."""
        probe = np.zeros(
            (max(8, int(self.input_h)), max(8, int(self.input_w)), 3),
            dtype=np.uint8,
        )
        prepared, _meta = self._prepare_input(probe)
        expected_outputs = set(str(name) for name in self._base.output_names)
        output_schema: Any = None

        run_count = max(1, int(probe_runs))
        for run_index in range(run_count):
            try:
                output = self._base.run_inference(prepared)
            except Exception as exc:
                raise RuntimeError(
                    f"[{self._base.name}] Hailo runtime health check failed on "
                    f"probe {run_index + 1}: {exc}"
                ) from exc

            if expected_outputs:
                if not isinstance(output, dict):
                    raise RuntimeError(
                        f"[{self._base.name}] Expected {len(expected_outputs)} output "
                        f"channels, received {type(output).__name__}."
                    )
                missing_outputs = sorted(expected_outputs.difference(output))
                if missing_outputs:
                    raise RuntimeError(
                        f"[{self._base.name}] Missing output channels after inference: "
                        + ", ".join(missing_outputs)
                    )
                for output_name in expected_outputs:
                    tensor = np.asarray(output[output_name])
                    if tensor.size == 0:
                        raise RuntimeError(
                            f"[{self._base.name}] Output channel {output_name} is empty."
                        )
            output_schema = self._describe_output(output)

        return {
            "name": self._base.name,
            "probe_runs": run_count,
            "last_inference_ms": float(
                getattr(self._base, "last_inference_ms", 0.0) or 0.0
            ),
            "output_schema": output_schema,
        }

    def _describe_output(self, output: Any):
        return self._module.HailoHEFModel._describe_output(self, output)

    def _get_attr_or_call(self, obj: Any, attr_name: str):
        return self._module.HailoHEFModel._get_attr_or_call(obj, attr_name)

    def _sigmoid(self, value: np.ndarray):
        return self._module.HailoHEFModel._sigmoid(value)

    def _softmax_last(self, value: np.ndarray):
        return self._module.HailoHEFModel._softmax_last(value)

    def _nms_indices(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float = 0.45,
        max_dets: int = 100,
    ):
        return self._module.HailoHEFModel._nms_indices(
            boxes,
            scores,
            iou_thresh=iou_thresh,
            max_dets=max_dets,
        )

    def _decode_yolov8_seg_raw(
        self,
        raw_output: Any,
        conf_thresh: float,
        iou_thresh: float,
        max_dets: int,
        include_masks: bool = True,
    ):
        return self._module.HailoHEFModel._decode_yolov8_seg_raw(
            self,
            raw_output,
            conf_thresh,
            iou_thresh,
            max_dets,
            include_masks=include_masks,
        )

    def _raw_score_peak(self, raw_output: Any):
        return self._module.HailoHEFModel._raw_score_peak(self, raw_output)

    def _extract_bbox_from_object(self, obj: Any):
        return self._module.HailoHEFModel._extract_bbox_from_object(self, obj)

    def _extract_mask_from_object(self, obj: Any):
        return self._module.HailoHEFModel._extract_mask_from_object(self, obj)

    def _extract_from_object(self, obj: Any, class_hint: int = 0):
        return self._module.HailoHEFModel._extract_from_object(self, obj, class_hint=class_hint)

    def _row_to_det(self, row: np.ndarray, class_hint: int = 0):
        return self._module.HailoHEFModel._row_to_det(self, row, class_hint=class_hint)

    def _extract_from_array(self, arr: np.ndarray, conf_thresh: float, class_hint: int = 0):
        return self._module.HailoHEFModel._extract_from_array(
            self, arr, conf_thresh, class_hint=class_hint
        )

    def _extract_raw_detections(self, raw_output: Any, conf_thresh: float):
        return self._module.HailoHEFModel._extract_raw_detections(self, raw_output, conf_thresh)

    def _map_bbox_to_frame(self, bbox, meta, frame_w: int, frame_h: int):
        return self._module.HailoHEFModel._map_bbox_to_frame(self, bbox, meta, frame_w, frame_h)

    def _map_mask_to_frame(self, mask: np.ndarray, meta, frame_shape, mask_thresh: float = 0.5):
        return self._module.HailoHEFModel._map_mask_to_frame(
            self, mask, meta, frame_shape, mask_thresh=mask_thresh
        )

    def predict(
        self,
        frame_bgr: np.ndarray,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        mask_thresh: float = 0.5,
        max_dets: int = 100,
        cancel_event: threading.Event | None = None,
        include_masks: bool = True,
    ):
        started_at = time.perf_counter()
        if cancel_event is not None:
            self._cancel_event = cancel_event
        result = self._module.HailoHEFModel.predict(
            self,
            frame_bgr,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            mask_thresh=mask_thresh,
            max_dets=max_dets,
            include_masks=include_masks,
        )
        total_ms = (time.perf_counter() - started_at) * 1000.0
        hardware_ms = float(getattr(self._base, "last_inference_ms", 0.0) or 0.0)
        postprocess_ms = max(0.0, total_ms - hardware_ms)
        if total_ms >= 1000.0:
            logger.warning(
                "[%s] Slow purity prediction: total=%.1fms, hailo=%.1fms, "
                "postprocess=%.1fms, detections=%s",
                self._base.name,
                total_ms,
                hardware_ms,
                postprocess_ms,
                len(result or []),
            )
        return result


class PurityTestManager:
    def __init__(
        self,
        *,
        base_dir: Path,
        frame_getter: FrameGetter,
        status_setter: StatusSetter | None = None,
        speak_fn: Callable[[str], None] | None = None,
        session_start_fn: SessionCallback | None = None,
        session_stop_fn: SessionCallback | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.frame_getter = frame_getter
        self.status_setter = status_setter or (lambda _message: None)
        self.speak_fn = speak_fn or (lambda _text: None)
        self.session_start_fn = session_start_fn or (lambda: None)
        self.session_stop_fn = session_stop_fn or (lambda: None)

        self._module: Any = None
        self._module_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._video_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._session_resource_lock = threading.Lock()

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._video_writer: cv2.VideoWriter | None = None
        self._last_video_write_at = 0.0
        self._display_frame: np.ndarray | None = None
        self._display_frame_updated_at = 0.0
        self._audio_worker: Any = None
        self._audio_bundle: tuple[Any, Any] | None = None
        self._models_loaded = False
        self._availability_error = ""
        self._selected_audio_device: str | int | None = "__AUTO__"
        self._audio_ok_confidence_threshold = 0.70
        self._session_root: Path | None = None
        self._session_resources_active = False
        self._last_error = ""
        self._consecutive_hailo_errors = 0
        self._session = PuritySessionSummary()

        self.model_dir = self.base_dir / "models"
        self.stone_model_path = self.model_dir / "yolov8s_seg.hef"
        self.run_local_path = self.base_dir / "run-local4.py"
        self._available = self.run_local_path.exists()
        self._last_stone_announced = False
        self._rubbing_started_announced = False
        self._acid_detected_announced = False
        self._last_loop_heartbeat_at = 0.0
        self._requested_stop_reason = ""

    def speak(self, text: str) -> None:
        try:
            self.speak_fn(text)
        except Exception:
            pass

    def _module_path(self) -> Path:
        return self.run_local_path

    def _ensure_module(self) -> Any:
        if self._module is not None:
            return self._module
        with self._module_lock:
            if self._module is not None:
                return self._module
            if not self._module_path().exists():
                raise FileNotFoundError(f"Missing purity pipeline script: {self._module_path()}")
            spec = importlib.util.spec_from_file_location("embedded_run_local4", str(self._module_path()))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to load module from {self._module_path()}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._module = module
            return module

    def _set_status(self, message: str) -> None:
        try:
            self.status_setter(message)
        except Exception:
            pass

    @staticmethod
    def _is_transient_hailo_error(message: str) -> bool:
        normalized = str(message or "").lower()
        return any(
            marker in normalized
            for marker in (
                "hailo_queue_is_full",
                "queue is full",
            )
        )

    def _handle_inference_error(self, message: str) -> bool:
        """Return True when a transient Hailo failure should be retried."""
        if not self._is_transient_hailo_error(message):
            return False

        self._consecutive_hailo_errors += 1
        if self._consecutive_hailo_errors >= MAX_CONSECUTIVE_HAILO_ERRORS:
            return False

        logger.warning(
            "Hailo inference queue was temporarily full; retrying purity inference "
            "(%s/%s): %s",
            self._consecutive_hailo_errors,
            MAX_CONSECUTIVE_HAILO_ERRORS,
            message,
        )
        self._set_status("Hailo inference queue was busy. Retrying the purity test...")
        with self._state_lock:
            self._session.status = "Hailo inference queue busy; retrying"
        return True

    def _mark_inference_started(self, stage: str, frame: np.ndarray) -> None:
        normalized_stage = str(stage or "PURITY").upper()
        with self._state_lock:
            self._session.inference_status = f"{normalized_stage} inference running"
        # Keep the most recent annotated frame visible while synchronous Hailo
        # inference is running. Publishing the raw frame here made the preview
        # alternate between annotated and unannotated images every cycle.

    def _mark_inference_success(self, stage: str, inference_ms: float = 0.0) -> None:
        self._consecutive_hailo_errors = 0
        normalized_stage = str(stage or "PURITY").upper()
        with self._state_lock:
            self._session.inference_status = f"{normalized_stage} inference active"
            self._session.inference_count += 1
            self._session.last_inference_stage = normalized_stage
            self._session.last_inference_ms = max(0.0, float(inference_ms or 0.0))
            self._session.last_inference_at = self._stamp()
            inference_count = int(self._session.inference_count)
            measured_ms = float(self._session.last_inference_ms)
        if inference_count == 1 or inference_count % 30 == 0:
            logger.info(
                "[PurityInference] stage=%s completed_frames=%s cycle_ms=%.1f",
                normalized_stage,
                inference_count,
                measured_ms,
            )
        if measured_ms >= 1000.0 and (
            inference_count <= 3 or inference_count % 10 == 0
        ):
            logger.warning(
                "[PurityInference] Slow %s frame cycle: %.1fms",
                normalized_stage,
                measured_ms,
            )

    def _draw_inference_banner(
        self,
        frame: np.ndarray,
        stage: str,
        status: str,
        count: int,
        inference_ms: float,
    ) -> np.ndarray:
        display = frame.copy()
        stage_text = str(stage or "PURITY").upper()
        status_text = str(status or "ACTIVE").upper()
        latency_text = f" | {inference_ms:.0f} ms" if inference_ms > 0.0 else ""
        label = f"{stage_text} INFERENCE {status_text} | frames {max(0, int(count))}{latency_text}"
        cv2.rectangle(display, (12, 12), (min(display.shape[1] - 12, 690), 54), (20, 20, 20), -1)
        cv2.putText(
            display,
            label,
            (24, 41),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 255),
            2,
        )
        return display

    def _activate_session_resources(self) -> None:
        with self._session_resource_lock:
            if self._session_resources_active:
                return
            try:
                self.session_start_fn()
            except Exception:
                try:
                    self.session_stop_fn()
                except Exception:
                    logger.exception("Could not restore resources after purity start failure")
                raise
            self._session_resources_active = True

    def _release_session_resources(self) -> None:
        with self._session_resource_lock:
            if not self._session_resources_active:
                return
            self._session_resources_active = False
            try:
                self.session_stop_fn()
            except Exception:
                logger.exception("Could not restore resources after purity test")

    def preload(self, runtime: HailoRuntime) -> dict[str, Any]:
        created_models: list[Any] = []
        try:
            module = self._ensure_module()
            self._wire_model_paths(module)
            self._set_status("Loading purity test models...")
            required_files = [
                self.run_local_path,
                self.stone_model_path,
                self.model_dir / "gold.hef",
                self.model_dir / "bestnewacid.hef",
            ]
            missing_files = [str(path) for path in required_files if not path.is_file()]
            if missing_files:
                raise FileNotFoundError(
                    "Missing purity test file(s): " + ", ".join(missing_files)
                )
            if not bool(getattr(module, "HAILO_AVAILABLE", False)):
                raise RuntimeError("hailo_platform is not available for the purity test models.")
            if not self._models_loaded:
                model_stone = runtime.create_model(
                    str(self.stone_model_path),
                    "PurityStone",
                    timeout_ms=PURITY_HAILO_INFERENCE_TIMEOUT_MS,
                    batch_size=DEFAULT_HAILO_BATCH_SIZE,
                )
                if model_stone is None:
                    detail = str(getattr(runtime, "last_model_error", "") or "").strip()
                    raise RuntimeError(
                        "Could not create the purity stone HEF model."
                        + (f" Last Hailo error: {detail}" if detail else "")
                    )
                created_models.append(model_stone)

                model_gold = runtime.create_model(
                    str(self.model_dir / "gold.hef"),
                    "PurityGold",
                    timeout_ms=PURITY_HAILO_INFERENCE_TIMEOUT_MS,
                    batch_size=DEFAULT_HAILO_BATCH_SIZE,
                )
                if model_gold is None:
                    detail = str(getattr(runtime, "last_model_error", "") or "").strip()
                    raise RuntimeError(
                        "Could not create the purity gold HEF model."
                        + (f" Last Hailo error: {detail}" if detail else "")
                    )
                created_models.append(model_gold)

                model_acid = runtime.create_model(
                    str(self.model_dir / "bestnewacid.hef"),
                    "PurityAcid",
                    timeout_ms=PURITY_HAILO_INFERENCE_TIMEOUT_MS,
                    batch_size=DEFAULT_HAILO_BATCH_SIZE,
                )
                if model_acid is None:
                    detail = str(getattr(runtime, "last_model_error", "") or "").strip()
                    raise RuntimeError(
                        "Could not create the purity acid HEF model."
                        + (f" Last Hailo error: {detail}" if detail else "")
                    )
                created_models.append(model_acid)
                module.MODEL_STONE = PurityHailoAdapter(model_stone, module)
                module.MODEL_GOLD = PurityHailoAdapter(model_gold, module)
                module.MODEL_ACID = PurityHailoAdapter(model_acid, module)
                for adapter in (module.MODEL_STONE, module.MODEL_GOLD, module.MODEL_ACID):
                    contract = adapter.validate_input_contract()
                    logger.info(
                        "[%s] Purity input contract OK: layout=%s shape=%s dtype=%s bytes=%s",
                        contract["name"],
                        contract["layout"],
                        contract["shape"],
                        contract["dtype"],
                        contract["bytes"],
                    )
                self._models_loaded = True
                created_models = []
            if self._audio_bundle is None and bool(getattr(module, "AUDIO_AVAILABLE", False)):
                self._set_status("Loading purity audio model...")
                self._audio_bundle = module.load_audio_model()
            self._availability_error = ""
            with self._state_lock:
                self._session.available = True
                self._session.models_loaded = bool(self._models_loaded)
                if self._audio_bundle is None and bool(getattr(module, "AUDIO_AVAILABLE", False)):
                    self._session.status = "Purity audio model unavailable. Visual-only fallback will be used."
        except Exception as exc:  # noqa: BLE001
            for model in created_models:
                try:
                    model.close()
                except Exception:
                    pass
                try:
                    runtime.models.remove(model)
                except (AttributeError, ValueError):
                    pass
            module = self._module
            if module is not None and not self._models_loaded:
                module.MODEL_STONE = None
                module.MODEL_GOLD = None
                module.MODEL_ACID = None
            logger.exception("Purity test preload failed")
            self._availability_error = str(exc)
            with self._state_lock:
                self._session.available = False
                self._session.models_loaded = False
                self._session.error = str(exc)
                self._session.status = f"Purity test unavailable: {exc}"
            self._set_status(f"Purity preload warning: {exc}")
        return self.snapshot()

    def validate_runtime_models(self, probe_runs: int = 1) -> list[dict[str, Any]]:
        if not self._models_loaded:
            raise RuntimeError("Purity models are not loaded.")
        module = self._ensure_module()
        results = []
        for adapter_name in ("MODEL_STONE", "MODEL_GOLD", "MODEL_ACID"):
            adapter = getattr(module, adapter_name, None)
            if adapter is None:
                raise RuntimeError(f"Missing loaded purity adapter: {adapter_name}")
            result = adapter.validate_runtime_contract(probe_runs=probe_runs)
            results.append(result)
            logger.info(
                "[%s] Startup runtime probe passed: runs=%s last_ms=%.1f",
                result["name"],
                result["probe_runs"],
                result["last_inference_ms"],
            )
        return results

    def unload_models(self, runtime: HailoRuntime | None) -> None:
        with self._state_lock:
            if self._session.running:
                raise RuntimeError("Cannot unload purity models while the test is running.")

        module = self._module
        adapters = []
        if module is not None:
            adapters = [
                getattr(module, "MODEL_STONE", None),
                getattr(module, "MODEL_GOLD", None),
                getattr(module, "MODEL_ACID", None),
            ]
            module.MODEL_STONE = None
            module.MODEL_GOLD = None
            module.MODEL_ACID = None

        closed_model_ids: set[int] = set()
        for adapter in adapters:
            model = getattr(adapter, "_base", adapter)
            if model is None or id(model) in closed_model_ids:
                continue
            closed_model_ids.add(id(model))
            try:
                model.close()
            except Exception:
                logger.exception("Could not close purity Hailo model")
            if runtime is not None:
                try:
                    runtime.models.remove(model)
                except (AttributeError, ValueError):
                    pass

        self._models_loaded = False
        with self._state_lock:
            self._session.models_loaded = False

    def _wire_model_paths(self, module: Any) -> None:
        module.BASE_DIR = str(self.base_dir)
        module.MODEL_GOLD_PATH = str(self.model_dir / "gold.hef")
        module.MODEL_STONE_PATH = str(self.stone_model_path)
        module.MODEL_ACID_PATH = str(self.model_dir / "bestnewacid.hef")
        rubbing_audio_dir = self.base_dir / "new_audio_rubbing" / "models"
        module.SOUND_MODEL_DIR = str(rubbing_audio_dir)
        module.SOUND_MODEL_PATH = str(rubbing_audio_dir / "gold_rub_cnn.tflite")
        module.AUDIO_CONF_THRESH = float(self._audio_ok_confidence_threshold)

    def list_audio_devices(self) -> list[dict[str, Any]]:
        try:
            module = self._ensure_module()
        except Exception:
            return []
        try:
            return list(module.list_audio_devices() or [])
        except Exception:
            return []

    def select_audio_device(self, value: str | int | None) -> None:
        module = self._module
        auto_token = getattr(module, "AUDIO_DEVICE_AUTO", "__AUTO__") if module is not None else "__AUTO__"
        if value is None:
            self._selected_audio_device = auto_token
            return
        if isinstance(value, int):
            self._selected_audio_device = value
            return
        raw = str(value).strip()
        if not raw:
            self._selected_audio_device = auto_token
            return
        if raw == auto_token:
            self._selected_audio_device = auto_token
            return
        try:
            self._selected_audio_device = int(raw)
        except Exception:
            self._selected_audio_device = raw

    def set_audio_ok_confidence_threshold(self, value: float) -> float:
        threshold = max(0.50, min(0.99, float(value)))
        self._audio_ok_confidence_threshold = threshold
        module = self._module
        if module is not None:
            try:
                module.AUDIO_CONF_THRESH = threshold
            except Exception:
                pass
        worker = self._audio_worker
        if worker is not None:
            try:
                threshold = float(worker.set_confidence_threshold(threshold))
            except Exception:
                logger.exception("Could not update active purity audio threshold")
        with self._state_lock:
            self._session.audio_ok_threshold = threshold
        return threshold

    def audio_ok_confidence_threshold(self) -> float:
        return float(self._audio_ok_confidence_threshold)

    def _current_audio_device_label(self) -> str:
        if self._selected_audio_device == "__AUTO__":
            return "Auto"
        if self._selected_audio_device is None:
            return "Default"
        return str(self._selected_audio_device)

    def _start_audio_worker(self, module: Any) -> None:
        self._stop_audio_worker()
        if self._audio_bundle is None or not bool(getattr(module, "AUDIO_AVAILABLE", False)):
            return
        audio_model, audio_device_ctx = self._audio_bundle
        selected_data = self._selected_audio_device
        allow_fallback = False
        auto_token = getattr(module, "AUDIO_DEVICE_AUTO", "__AUTO__")
        if selected_data == auto_token:
            selected_data = module.find_preferred_audio_input_device()
            allow_fallback = True
        elif selected_data is None:
            allow_fallback = True
        else:
            selected_data = module.resolve_audio_input_device(selected_data)
        worker = module.AudioWorker(
            audio_model,
            audio_device_ctx,
            confidence_threshold=self._audio_ok_confidence_threshold,
            device=selected_data,
            allow_fallback=allow_fallback,
        )
        worker.start()
        try:
            stream_open = bool((worker.get_debug_snapshot() or {}).get("stream_open", False))
        except Exception:
            stream_open = False
        if not stream_open:
            try:
                worker.stop()
            except Exception:
                pass
            logger.warning(
                "[Audio] No microphone stream opened; purity test will use visual-only fallback."
            )
            return
        self._audio_worker = worker

    def _stop_audio_worker(self) -> None:
        worker = self._audio_worker
        self._audio_worker = None
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass

    def _get_audio_debug_snapshot(self) -> dict[str, Any]:
        worker = self._audio_worker
        if worker is None:
            return {}
        try:
            return worker.get_debug_snapshot() or {}
        except Exception:
            return {}

    def _get_audio_debug_text(self) -> tuple[str, str, str]:
        if self._audio_worker is None:
            if self._audio_bundle is None:
                return ("", "Audio unavailable", "Audio dependencies or model are unavailable.")
            return ("", "Audio worker stopped", "Select a microphone and start the purity test.")
        try:
            debug = self._audio_worker.get_debug_snapshot() or {}
        except Exception:
            return ("", "Audio debug unavailable", "Could not read audio worker state.")
        label = str(debug.get("last_label", "Waiting...") or "Waiting...")
        conf = float(debug.get("last_conf", 0.0) or 0.0)
        decision = str(debug.get("last_decision", "") or "")
        prediction = f"{label} ({conf:.2f})" + (f" -> {decision}" if decision else "")
        device_name = str(debug.get("selected_device_name", "") or "")
        probabilities = debug.get("probabilities", {}) or {}
        top_probs = sorted(
            ((str(name), float(prob)) for name, prob in probabilities.items()),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        probability_text = ", ".join(f"{name}={prob:.2f}" for name, prob in top_probs) or "no probabilities"
        debug_text = (
            f"mic={device_name or 'unknown'} | "
            f"decision={decision or 'Waiting...'} | "
            f"{probability_text} | "
            f"rms={float(debug.get('rms', 0.0) or 0.0):.4f} | "
            f"peak={float(debug.get('peak', 0.0) or 0.0):.4f} | "
            f"gain={debug.get('capture_gain', 'unknown') or 'unknown'} | "
            f"active={float(debug.get('active_ratio', 0.0) or 0.0):.3f} | "
            f"sr={int(debug.get('input_sr', 0) or 0)}/{int(debug.get('model_sr', 0) or 0)} | "
            f"backend={debug.get('model_backend', '') or 'unknown'} | "
            f"stream={'open' if bool(debug.get('stream_open', False)) else 'closed'}"
        )
        return (device_name, prediction, debug_text)

    def _set_display_frame(self, frame: np.ndarray | None) -> None:
        with self._display_lock:
            self._display_frame = None if frame is None else frame.copy()
            self._display_frame_updated_at = time.monotonic() if frame is not None else 0.0

    def get_display_frame_copy(self, max_age_s: float | None = None) -> np.ndarray | None:
        with self._display_lock:
            if self._display_frame is None:
                return None
            if (
                max_age_s is not None
                and self._display_frame_updated_at > 0.0
                and time.monotonic() - self._display_frame_updated_at > max(0.0, max_age_s)
            ):
                return None
            return self._display_frame.copy()

    def is_running(self) -> bool:
        with self._state_lock:
            return bool(self._session.running)

    def worker_is_active(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def build_live_preview(self, frame: np.ndarray) -> np.ndarray:
        """Decorate a fresh camera frame while synchronous inference is busy."""
        display = frame.copy()
        module = self._module
        if module is not None:
            try:
                display = module.draw_status(display)
            except Exception:
                pass
        with self._state_lock:
            stage = self._session.last_inference_stage or self._session.stage
            count = int(self._session.inference_count)
            inference_ms = float(self._session.last_inference_ms)
            status = (
                "RUNNING"
                if "running" in self._session.inference_status.lower()
                else "LIVE"
            )
        return self._draw_inference_banner(
            display,
            stage,
            status,
            count,
            inference_ms,
        )

    def _ensure_video_writer(self, frame: np.ndarray) -> None:
        with self._video_lock:
            if self._video_writer is not None or self._session_root is None:
                return
            output_path = self._session_root / "session_recording.avi"
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                PURITY_VIDEO_WRITE_FPS,
                (int(width), int(height)),
            )
            if writer is not None and writer.isOpened():
                self._video_writer = writer
                with self._state_lock:
                    self._session.video_path = str(output_path)

    def _write_video_frame(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        now = time.monotonic()
        min_interval = 1.0 / PURITY_VIDEO_WRITE_FPS
        if now - self._last_video_write_at < min_interval:
            return
        self._ensure_video_writer(frame)
        with self._video_lock:
            if self._video_writer is not None:
                try:
                    self._video_writer.write(frame)
                    self._last_video_write_at = now
                except Exception:
                    pass

    def _release_video_writer(self) -> None:
        with self._video_lock:
            writer = self._video_writer
            self._video_writer = None
            self._last_video_write_at = 0.0
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

    def _save_frame(self, path: Path, frame: np.ndarray | None) -> str:
        if frame is None:
            return ""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if cv2.imwrite(str(path), frame):
                return str(path)
        except Exception:
            pass
        return ""

    def _capture_session_image(self, field_name: str, prefix: str, frame: np.ndarray | None) -> str:
        if self._session_root is None:
            return ""
        filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        saved_path = self._save_frame(self._session_root / filename, frame)
        if saved_path:
            with self._state_lock:
                setattr(self._session, field_name, saved_path)
        return saved_path

    def _save_zoom_region_image(
        self,
        field_name: str,
        prefix: str,
        frame: np.ndarray | None,
        bbox: tuple[int, int, int, int] | None,
        mask: np.ndarray | None = None,
    ) -> str:
        if frame is None or bbox is None:
            return ""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            return ""
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        pad = max(4, int(max(x2 - x1, y2 - y1) * 0.45))
        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(w, x2 + pad)
        ry2 = min(h, y2 + pad)
        crop = frame[ry1:ry2, rx1:rx2].copy()
        if crop.size == 0:
            return ""
        if mask is not None:
            try:
                mask_crop = mask[ry1:ry2, rx1:rx2]
                if mask_crop.shape[:2] == crop.shape[:2]:
                    softened = cv2.GaussianBlur((mask_crop > 0).astype(np.uint8) * 255, (0, 0), 2.0)
                    softened_f = np.clip(softened.astype(np.float32) / 255.0, 0.25, 1.0)[..., None]
                    crop = np.clip(crop.astype(np.float32) * softened_f, 0, 255).astype(np.uint8)
            except Exception:
                pass
        return self._capture_session_image(field_name, prefix, crop)

    def _stamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _build_progress_snapshot(self, session: dict[str, Any]) -> dict[str, Any]:
        module = self._module
        stage = str(session.get("stage", "IDLE") or "IDLE").upper()
        running = bool(session.get("running"))
        result = str(session.get("result", "") or "")
        audio_mode = str(session.get("audio_mode", "sync") or "sync").lower()
        module_state = getattr(module, "STATE", {}) if module is not None else {}
        now_ts = time.time()

        def item(key: str, label: str, level: float, status: str, tone: str, *, done: bool = False):
            normalized = max(0.0, min(1.0, float(level)))
            return {
                "key": key,
                "label": label,
                "level": normalized,
                "percent": int(round(normalized * 100.0)),
                "status": str(status or ""),
                "tone": str(tone or "idle"),
                "done": bool(done),
            }

        inference_count = int(session.get("inference_count", 0) or 0)
        inference_status = str(session.get("inference_status", "Idle") or "Idle")
        if running and inference_count > 0:
            inference_item = item(
                "inference",
                "Vision inference",
                1.0,
                f"Active ({inference_count} frames)",
                "success",
            )
        elif running:
            inference_item = item(
                "inference",
                "Vision inference",
                0.35,
                inference_status,
                "warn",
            )
        elif session.get("error"):
            inference_item = item("inference", "Vision inference", 0.0, "Inference error", "danger")
        else:
            inference_item = item("inference", "Vision inference", 0.0, inference_status, "idle")

        stone_bbox = module_state.get("last_stone_bbox")
        stone_done = bool(session.get("rubbing_ok") or stage in {"ACID", "COMPLETED"} or session.get("acid_ok"))
        if stone_done:
            stone_item = item("stone", "Stone", 1.0, "Stone OK", "done", done=True)
        elif stone_bbox is not None:
            stone_item = item("stone", "Stone", 1.0, "Stone visible", "success")
        else:
            stone_item = item("stone", "Stone", 0.0, "Waiting for stone", "danger" if running else "idle")

        gold_visible = bool(module_state.get("gold_visible_now", False))
        gold_recent_until = float(module_state.get("gold_detected_recent_until", 0.0) or 0.0)
        gold_recent = now_ts <= gold_recent_until
        if stone_done:
            gold_item = item("gold", "Gold", 1.0, "Gold OK", "done", done=True)
        elif gold_visible:
            gold_item = item("gold", "Gold", 1.0, "Gold detected", "success")
        elif gold_recent:
            gold_item = item("gold", "Gold", 0.65, "Recent gold lock", "warn")
        else:
            gold_item = item("gold", "Gold", 0.0, "Waiting for gold", "danger" if running else "idle")

        sound_status = str(session.get("sound_status", "Waiting...") or "Waiting...")
        audio_debug = self._get_audio_debug_snapshot()
        stream_open = bool(audio_debug.get("stream_open", False))
        active_ratio = float(audio_debug.get("active_ratio", 0.0) or 0.0)
        last_label = str(
            session.get("audio_label")
            or audio_debug.get("last_label", "Waiting...")
            or "Waiting..."
        )
        last_decision = str(
            session.get("audio_decision")
            or audio_debug.get("last_decision", "")
            or ""
        )
        if audio_mode == "visual-only":
            audio_item = item("audio", "Audio", 1.0, "Visual-only mode", "warn")
        elif sound_status == "OK":
            audio_item = item("audio", "Audio", 1.0, f"{last_label} OK", "success", done=True)
        elif stream_open:
            heard = active_ratio >= 0.02 or last_label not in {"", "Waiting..."}
            audio_item = item(
                "audio",
                "Audio",
                0.6 if heard else 0.25,
                f"{last_label} {last_decision}".strip() if heard else "Waiting for rubbing sound",
                "warn" if heard else "danger",
            )
        else:
            audio_item = item("audio", "Audio", 0.0, "Microphone not ready", "danger" if running else "idle")

        rubbing_confirm_frames = int(getattr(module, "RUBBING_SYNC_CONFIRM_FRAMES", 2) or 2) if module is not None else 2
        rubbing_hits = int(module_state.get("rubbing_sync_hits", 0) or 0)
        visual_recent_until = float(module_state.get("visual_rubbing_recent_until", 0.0) or 0.0)
        audio_recent_until = float(module_state.get("audio_ok_recent_until", 0.0) or 0.0)
        visual_recent = now_ts <= visual_recent_until
        audio_recent = now_ts <= audio_recent_until and sound_status == "OK"
        rubbing_done = bool(session.get("rubbing_ok") or stage in {"ACID", "COMPLETED"} or session.get("acid_ok"))
        if rubbing_done:
            rubbing_item = item("rubbing", "Visual / Rubbing", 1.0, "Visual OK", "done", done=True)
        elif running and stage == "RUBBING":
            sync_ready = visual_recent and (audio_mode == "visual-only" or audio_recent)
            if sync_ready:
                progress = max(0.7, min(1.0, rubbing_hits / float(max(1, rubbing_confirm_frames))))
                rubbing_item = item("rubbing", "Visual / Rubbing", progress, "Visual OK", "success")
            else:
                rubbing_item = item("rubbing", "Visual / Rubbing", 0.3, "Waiting for sync", "warn")
        else:
            rubbing_item = item("rubbing", "Visual / Rubbing", 0.0, "Waiting", "idle")

        acid_confirm_frames = int(getattr(module, "ACID_CONFIRM_FRAMES", 3) or 3) if module is not None else 3
        acid_streak = int(module_state.get("acid_positive_streak", 0) or 0)
        acid_done = bool(session.get("acid_ok") or stage == "COMPLETED" or result == "SUCCESS")
        if acid_done:
            acid_item = item("acid", "Acid", 1.0, "Acid OK", "done", done=True)
        elif stage == "ACID":
            acid_level = max(0.28, min(1.0, acid_streak / float(max(1, acid_confirm_frames))))
            acid_item = item("acid", "Acid", acid_level, "Acid test running", "success" if acid_level >= 1.0 else "warn")
        elif rubbing_done:
            acid_item = item("acid", "Acid", 0.1, "Ready to start", "idle")
        else:
            acid_item = item("acid", "Acid", 0.0, "Waiting for rubbing", "locked")

        return {"items": [inference_item, stone_item, gold_item, audio_item, rubbing_item, acid_item]}

    def start(self, session_root: Path, audio_device: str | int | None = None) -> dict[str, Any]:
        module = self._ensure_module()
        if not self._models_loaded:
            raise RuntimeError(self._availability_error or "Purity models are not loaded.")
        if audio_device is not None:
            self.select_audio_device(audio_device)
        with self._state_lock:
            already_running = bool(self._session.running)
        if already_running:
            return self.snapshot()
        session_initialized = False
        try:
            self._activate_session_resources()
            initial_frame = self.frame_getter()
            if initial_frame is None:
                raise RuntimeError("The camera did not provide a frame for the purity test.")
            logger.info(
                "[PurityFrame] inference input frame=%sx%s",
                int(initial_frame.shape[1]),
                int(initial_frame.shape[0]),
            )
            logger.info(
                "[PurityConfig] infer_skip=%s timeout_ms=%s "
                "stone_conf=%.2f stone_min_area=%.3f "
                "gold_conf=%.2f gold_inside=%.2f gold_full_fallback_every=%s "
                "rubbing_confirm=%s acid_conf=%.2f acid_confirm=%s",
                max(1, int(getattr(module, "INFER_SKIP", 1) or 1)),
                PURITY_HAILO_INFERENCE_TIMEOUT_MS,
                float(getattr(module, "STONE_CONF_THRESH", 0.0) or 0.0),
                float(getattr(module, "STONE_MIN_AREA_RATIO", 0.0) or 0.0),
                float(getattr(module, "GOLD_CONF_THRESH", 0.0) or 0.0),
                float(getattr(module, "GOLD_MIN_INSIDE_STONE_RATIO", 0.0) or 0.0),
                int(getattr(module, "GOLD_FULL_FRAME_FALLBACK_EVERY", 1) or 1),
                int(getattr(module, "RUBBING_SYNC_CONFIRM_FRAMES", 1) or 1),
                float(getattr(module, "ACID_CONF_THRESH", 0.0) or 0.0),
                int(getattr(module, "ACID_CONFIRM_FRAMES", 1) or 1),
            )

            module.reset_state()
            self._stop_event.clear()
            self._last_error = ""
            self._consecutive_hailo_errors = 0
            self._requested_stop_reason = ""
            # Wire the stop event into every purity adapter for cancellation support
            for adapter_name in ("MODEL_STONE", "MODEL_GOLD", "MODEL_ACID"):
                adapter = getattr(module, adapter_name, None)
                if adapter is not None:
                    adapter._cancel_event = self._stop_event
            self._session_root = Path(session_root)
            self._session_root.mkdir(parents=True, exist_ok=True)
            self._last_stone_announced = False
            self._rubbing_started_announced = False
            self._acid_detected_announced = False
            self._last_loop_heartbeat_at = 0.0
            with self._state_lock:
                self._session = PuritySessionSummary(
                    available=True,
                    models_loaded=True,
                    running=True,
                    started_at=self._stamp(),
                    stage="RUBBING",
                    result="RUNNING",
                    status="Purity test running",
                    audio_mode="sync",
                    audio_selected_device=self._current_audio_device_label(),
                    audio_selected_device_value=self._selected_audio_device,
                    sound_status="Waiting...",
                    audio_ok_threshold=self._audio_ok_confidence_threshold,
                    inference_status="RUBBING inference starting",
                )
            session_initialized = True

            self._start_audio_worker(module)
            if self._audio_bundle is None or self._audio_worker is None:
                with self._state_lock:
                    self._session.audio_mode = "visual-only"
                    self._session.status = "Purity test running (visual-only fallback)"

            self._set_display_frame(
                self._draw_inference_banner(
                    initial_frame,
                    "RUBBING",
                    "STARTING",
                    0,
                    0.0,
                )
            )
            self.speak("Acid test has been started, keep the rubbing stone inside the camera feed.")
            self._set_status("Purity test started.")
            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="purity-test-loop",
            )
            self._thread.start()
        except Exception as exc:
            logger.exception("Could not start purity test")
            if session_initialized:
                self._finalize(error_text=str(exc))
            else:
                self._release_session_resources()
            raise
        return self.snapshot()

    def stop(self, reason: str = "Stopped by user") -> dict[str, Any]:
        self._stop_event.set()
        self._requested_stop_reason = str(reason or "Stopped by user")
        self._stop_audio_worker()
        with self._state_lock:
            if self._session.running:
                self._session.status = "Stopping purity test..."
                self._session.inference_status = "Stopping"
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(
                timeout=max(
                    1.0,
                    PURITY_HAILO_INFERENCE_TIMEOUT_MS / 1000.0 + 1.0,
                )
            )
        if thread is None or not thread.is_alive():
            self._release_session_resources()
        else:
            logger.error(
                "Purity worker did not stop before its Hailo inference timeout."
            )
        return self.snapshot()

    def _run_loop(self) -> None:
        module = self._ensure_module()
        frame_count = 0
        last_annotated: np.ndarray | None = None
        error_text = ""
        try:
            while not self._stop_event.is_set():
                raw_frame = self.frame_getter()
                if raw_frame is None:
                    time.sleep(0.03)
                    continue
                if self._stop_event.is_set():
                    break
                now_monotonic = time.monotonic()
                if now_monotonic - self._last_loop_heartbeat_at >= 5.0:
                    logger.info(
                        "[PurityLoop] alive stage=%s frame=%sx%s infer_skip=%s",
                        str(module.STATE.get("stage", "RUBBING") or "RUBBING"),
                        int(raw_frame.shape[1]),
                        int(raw_frame.shape[0]),
                        max(1, int(getattr(module, "INFER_SKIP", 1) or 1)),
                    )
                    self._last_loop_heartbeat_at = now_monotonic
                frame = raw_frame.copy()
                frame_count += 1
                if frame_count % max(1, int(getattr(module, "INFER_SKIP", 1) or 1)) == 0:
                    previous_stage = str(module.STATE.get("stage", "RUBBING") or "RUBBING")
                    if not bool(module.STATE.get("rubbing_done", False)):
                        self._mark_inference_started("RUBBING", raw_frame)
                        cycle_started_at = time.perf_counter()
                        annotated, info = module.process_rubbing_frame(frame)
                        rubbing_cycle_ms = (
                            time.perf_counter() - cycle_started_at
                        ) * 1000.0
                        if self._stop_event.is_set():
                            break
                        if info.get("error"):
                            inference_error = str(info["error"])
                            if self._handle_inference_error(inference_error):
                                module.STATE["rubbing_sync_hits"] = 0
                                last_annotated = frame.copy()
                                time.sleep(0.1)
                                continue
                            raise RuntimeError(inference_error)
                        self._mark_inference_success("RUBBING", rubbing_cycle_ms)

                        # Voice Command 6: Now Rubbing stone is detected...
                        if not self._last_stone_announced and module.STATE.get("stone_visible_now"):
                            self.speak("Now Rubbing stone is detected, use the jewelry to run on it")
                            self._last_stone_announced = True

                        # Voice Command 7: Jewelry is now inside the stone region...
                        if not self._rubbing_started_announced and module.STATE.get("gold_visible_now"):
                            self.speak("Jewelry is now inside the stone region, now start rubbing for acid test")
                            self._rubbing_started_announced = True

                        annotated, is_rubbing = module.compute_rubbing(annotated, info)

                        if self._audio_worker is not None:
                            combined_sync_ok, _visual_recent, _audio_recent = module.rubbing_sync_ready(is_rubbing, info)
                        else:
                            module.update_visual_rubbing_grace(is_rubbing, info)
                            combined_sync_ok = bool(is_rubbing)
                        if combined_sync_ok:
                            module.STATE["rubbing_sync_hits"] += 1
                        else:
                            module.STATE["rubbing_sync_hits"] = 0
                        if module.STATE["rubbing_sync_hits"] >= int(module.RUBBING_SYNC_CONFIRM_FRAMES):
                            module.STATE["rubbing_done"] = True
                            module.STATE["stage"] = "ACID"
                            module.STATE["acid_positive_streak"] = 0
                            module.STATE["rubbing_sync_hits"] = 0
                            annotated_display = annotated.copy()
                            self._capture_session_image("rubbing_image_path", "rubbing_ok", annotated_display)
                            self._save_zoom_region_image(
                                "rubbing_zoom_image_path",
                                "rubbing_zoom",
                                frame,
                                module.STATE.get("last_rubbing_bbox"),
                                mask=module.STATE.get("last_rubbing_mask"),
                            )
                            self._capture_session_image("acid_stage_image_path", "acid_stage", annotated_display)
                            self._set_status("Purity rubbing confirmed. Acid test running.")
                            self.speak("Visual and audio synchronization is okay. Now, apply the acid to complete the purity test.")
                        last_annotated = annotated.copy()
                    else:
                        self._mark_inference_started("ACID", raw_frame)
                        cycle_started_at = time.perf_counter()
                        annotated, acid_detected, acid_info = module.process_acid_frame(frame)
                        acid_cycle_ms = (
                            time.perf_counter() - cycle_started_at
                        ) * 1000.0
                        if self._stop_event.is_set():
                            break
                        if acid_info.get("error"):
                            inference_error = str(acid_info["error"])
                            if self._handle_inference_error(inference_error):
                                module.STATE["acid_positive_streak"] = 0
                                last_annotated = frame.copy()
                                time.sleep(0.1)
                                continue
                            raise RuntimeError(inference_error)
                        self._mark_inference_success(
                            "ACID",
                            acid_cycle_ms,
                        )
                        if acid_detected:
                            module.STATE["acid_positive_streak"] += 1
                        else:
                            module.STATE["acid_positive_streak"] = 0
                        if module.STATE["acid_positive_streak"] >= int(module.ACID_CONFIRM_FRAMES):
                            module.STATE["stage"] = "COMPLETED"
                            annotated_display = annotated.copy()
                            self._capture_session_image("acid_success_image_path", "acid_ok", annotated_display)
                            self._save_zoom_region_image(
                                "acid_zoom_image_path",
                                "acid_zoom",
                                frame,
                                acid_info.get("acid_bbox") or module.STATE.get("last_acid_bbox"),
                            )
                            self._capture_session_image("final_image_path", "final_frame", annotated_display)
                            self._set_status("Purity acid test completed successfully. Press Stop to continue.")
                            if not self._acid_detected_announced:
                                self.speak("Acid detected. Purity test completed. Click Stop to continue.")
                                self._acid_detected_announced = True
                        last_annotated = annotated.copy()

                    current_stage = str(module.STATE.get("stage", previous_stage) or previous_stage)
                    if previous_stage != current_stage and current_stage == "ACID":
                        with self._state_lock:
                            self._session.rubbing_ok = True
                    if current_stage == "COMPLETED":
                        with self._state_lock:
                            self._session.acid_ok = True
                            self._session.completed_at = self._session.completed_at or self._stamp()
                            self._session.result = "SUCCESS"
                elif last_annotated is None:
                    last_annotated = frame.copy()

                annotated_display = last_annotated if last_annotated is not None else frame
                base_display = annotated_display.copy()
                display = module.draw_status(base_display.copy())
                with self._state_lock:
                    inference_stage = self._session.last_inference_stage or self._session.stage
                    inference_count = int(self._session.inference_count)
                    inference_ms = float(self._session.last_inference_ms)
                display = self._draw_inference_banner(
                    display,
                    inference_stage,
                    "ACTIVE",
                    inference_count,
                    inference_ms,
                )
                self._write_video_frame(display)
                self._set_display_frame(display)

                audio_device_name, audio_prediction, audio_debug = self._get_audio_debug_text()
                audio_debug_snapshot = self._get_audio_debug_snapshot()
                audio_probabilities = audio_debug_snapshot.get("probabilities", {}) or {}
                with self._state_lock:
                    self._session.stage = str(module.STATE.get("stage", "IDLE") or "IDLE")
                    self._session.sound_status = str(module.STATE.get("sound_status", "Waiting...") or "Waiting...")
                    self._session.audio_prediction = audio_prediction
                    self._session.audio_label = str(
                        audio_debug_snapshot.get("last_label")
                        or module.STATE.get("audio_label")
                        or "Waiting..."
                    )
                    self._session.audio_decision = str(
                        audio_debug_snapshot.get("last_decision")
                        or module.STATE.get("audio_decision")
                        or self._session.sound_status
                        or "Waiting..."
                    )
                    self._session.audio_confidence = float(
                        audio_debug_snapshot.get("last_conf")
                        or module.STATE.get("audio_confidence")
                        or 0.0
                    )
                    self._session.audio_ok_threshold = float(
                        audio_debug_snapshot.get("threshold")
                        or self._audio_ok_confidence_threshold
                    )
                    self._session.audio_probabilities = {
                        str(name): float(prob)
                        for name, prob in audio_probabilities.items()
                    }
                    self._session.audio_input_rate = int(audio_debug_snapshot.get("input_sr", 0) or 0)
                    self._session.audio_model_rate = int(audio_debug_snapshot.get("model_sr", 0) or 0)
                    self._session.audio_model_backend = str(audio_debug_snapshot.get("model_backend", "") or "")
                    self._session.audio_device_name = audio_device_name
                    self._session.audio_debug = audio_debug
                    if self._session.result == "SUCCESS":
                        self._session.status = "Acid test done. Press Stop to unlock the final result."
                    elif self._session.stage == "ACID":
                        self._session.status = "Acid test running"
                    elif self._session.stage == "RUBBING":
                        self._session.status = (
                            "Rubbing test running (visual-only fallback)"
                            if self._session.audio_mode == "visual-only"
                            else "Rubbing test running"
                        )
                time.sleep(0.02)
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            self._last_error = error_text
            logger.exception("Purity test loop failed")
            self._set_status(f"Purity test error: {error_text}")
        finally:
            self._finalize(error_text=error_text)

    def _finalize(self, error_text: str = "") -> None:
        self._stop_event.set()
        self._stop_audio_worker()
        self._release_video_writer()
        self._set_display_frame(None)
        with self._state_lock:
            if self._session.started_at:
                self._session.running = False
                self._session.stopped_at = self._session.stopped_at or self._stamp()
                if not self._session.final_image_path:
                    self._session.final_image_path = self._session.acid_success_image_path or self._session.acid_stage_image_path
                if error_text:
                    self._session.error = error_text
                    self._session.result = "ERROR"
                    self._session.status = "Purity test stopped due to an error"
                    self._session.inference_status = "Inference error"
                elif self._session.acid_ok:
                    self._session.result = "SUCCESS"
                    self._session.status = "Purity test stopped after success"
                    self._session.inference_status = "Completed"
                elif self._session.rubbing_ok:
                    self._session.result = "Stopped after rubbing"
                    self._session.status = "Purity test stopped after rubbing stage"
                    self._session.inference_status = "Stopped"
                else:
                    self._session.result = self._requested_stop_reason or "Stopped by user"
                    self._session.status = "Purity test stopped"
                    self._session.inference_status = "Stopped"
        self._thread = None
        self._release_session_resources()

    def reset(self, stop_running: bool = True) -> None:
        if stop_running:
            self.stop("Reset")
            if self.worker_is_active():
                raise RuntimeError(
                    "Purity test is still stopping; wait before resetting the workflow."
                )
        with self._state_lock:
            self._session = PuritySessionSummary(
                available=self._available and not bool(self._availability_error),
                models_loaded=bool(self._models_loaded),
                error=self._availability_error,
                audio_ok_threshold=self._audio_ok_confidence_threshold,
                status="Purity test idle" if not self._availability_error else f"Purity test unavailable: {self._availability_error}",
            )
        self._session_root = None
        self._set_display_frame(None)

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            data = self._session.to_dict()
        data["available"] = bool(self._available and not self._availability_error)
        data["models_loaded"] = bool(self._models_loaded)
        data["last_error"] = self._last_error or self._availability_error
        data["processing_roi"] = None
        data["audio_ok_threshold"] = float(self._audio_ok_confidence_threshold)
        data["ui_progress"] = self._build_progress_snapshot(data)
        return data

    def shutdown(self) -> None:
        try:
            self.stop("Shutdown")
        except Exception:
            pass
