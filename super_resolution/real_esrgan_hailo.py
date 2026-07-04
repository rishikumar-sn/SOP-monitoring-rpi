from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

try:
    from hailo_platform import FormatType
except ImportError:  # pragma: no cover - exercised only without HailoRT installed
    FormatType = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HEF_PATH = SCRIPT_DIR / "real_esrgan_x2.hef"
MODEL_INPUT_SIZE = 512
MODEL_SCALE = 2
TILE_PAD = 64
TILE_STRIDE = MODEL_INPUT_SIZE - 2 * TILE_PAD


def tile_grid(width: int, height: int) -> tuple[int, int]:
    cols = max(1, math.ceil((width - TILE_STRIDE) / TILE_STRIDE) + 1)
    rows = max(1, math.ceil((height - TILE_STRIDE) / TILE_STRIDE) + 1)
    return cols, rows


class RealESRGANHailoX2:
    """Real-ESRGAN x2 runner for the Hailo HEF used by the standalone GUI."""

    def __init__(
        self,
        vdevice,
        hef_path: str | Path = DEFAULT_HEF_PATH,
        inference_lock: threading.Lock | threading.RLock | None = None,
        timeout_ms: int = 60000,
        allow_manual_activation: bool = True,
    ) -> None:
        if FormatType is None:
            raise RuntimeError("hailo_platform is not installed or could not be imported.")
        self.hef_path = Path(hef_path)
        if not self.hef_path.is_file():
            raise FileNotFoundError(f"Real-ESRGAN HEF not found: {self.hef_path}")
        if vdevice is None:
            raise RuntimeError("A configured Hailo VDevice is required for Real-ESRGAN.")

        self.timeout_ms = max(10000, int(timeout_ms))
        self.allow_manual_activation = bool(allow_manual_activation)
        self._inference_lock = inference_lock or threading.Lock()
        self._input_buffer = np.zeros(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
            dtype=np.uint8,
        )
        self._output_buffer = np.zeros(
            (
                MODEL_INPUT_SIZE * MODEL_SCALE,
                MODEL_INPUT_SIZE * MODEL_SCALE,
                3,
            ),
            dtype=np.uint8,
        )

        self.model = vdevice.create_infer_model(str(self.hef_path))
        self.model.input().set_format_type(FormatType.UINT8)
        try:
            self.model.output().set_format_type(FormatType.UINT8)
        except Exception:
            pass
        self._config_context = self.model.configure()
        self.configured_model = (
            self._config_context.__enter__()
            if hasattr(self._config_context, "__enter__")
            else self._config_context
        )
        self.bindings = self.configured_model.create_bindings()
        self.bindings.output().set_buffer(self._output_buffer)
        self._supports_async = hasattr(self.configured_model, "run_async")
        self._activation_supported: bool | None = None
        self._run_async_current = False
        self.last_runtime_seconds = 0.0
        self.last_tile_count = 0

    def close(self) -> None:
        try:
            self.configured_model.deactivate()
        except Exception:
            pass
        for attr in ("bindings", "configured_model", "model"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass
        if getattr(self, "_config_context", None) is not None and hasattr(
            self._config_context,
            "__exit__",
        ):
            try:
                self._config_context.__exit__(None, None, None)
            except Exception:
                pass
        self._config_context = None

    def process_bgr(
        self,
        image_bgr: np.ndarray,
        on_tile: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot super-resolve an empty image.")
        source = np.asarray(image_bgr)
        if source.ndim != 3 or source.shape[2] != 3:
            raise ValueError(f"Expected BGR image with shape HxWx3; received {source.shape}.")
        source = np.ascontiguousarray(source, dtype=np.uint8)

        started_at = time.perf_counter()
        height, width = source.shape[:2]
        with self._inference_lock:
            activated = False
            if self.allow_manual_activation and self._activation_supported is not False:
                try:
                    self.configured_model.activate()
                    activated = True
                    self._activation_supported = True
                except Exception:
                    if not self._supports_async:
                        raise
                    self._activation_supported = False
            if not activated and not self._supports_async:
                raise RuntimeError("This Hailo runtime supports neither manual activation nor async inference.")
            self._run_async_current = not activated
            try:
                try:
                    if width <= MODEL_INPUT_SIZE and height <= MODEL_INPUT_SIZE:
                        self.last_tile_count = 1
                        if on_tile:
                            on_tile(1, 1)
                        result = self._run_single(source)
                    else:
                        cols, rows = tile_grid(width, height)
                        self.last_tile_count = cols * rows
                        result = self._run_tiled(source, on_tile=on_tile)
                finally:
                    if activated:
                        try:
                            self.configured_model.deactivate()
                        except Exception:
                            pass
            except Exception as exc:
                raise RuntimeError(
                    f"Real-ESRGAN x2 Hailo inference failed for "
                    f"{width}x{height} image using {self.hef_path.name}: {exc}"
                ) from exc
            finally:
                self._run_async_current = False
        self.last_runtime_seconds = time.perf_counter() - started_at
        return result

    def self_test(self) -> dict[str, object]:
        probe = np.zeros((32, 32, 3), dtype=np.uint8)
        output = self.process_bgr(probe)
        return {
            "hef_path": str(self.hef_path),
            "input_shape": [32, 32, 3],
            "output_shape": [int(value) for value in output.shape],
            "tile_count": int(self.last_tile_count),
            "runtime_seconds": round(float(self.last_runtime_seconds), 3),
        }

    def _infer_tile_rgb(self, tile_rgb: np.ndarray) -> np.ndarray:
        self._input_buffer[:] = tile_rgb
        self.bindings.input().set_buffer(self._input_buffer)
        if self._run_async_current:
            completion_holder: dict[str, object] = {"info": None}

            def _callback(completion_info) -> None:
                completion_holder["info"] = completion_info

            self.configured_model.wait_for_async_ready(timeout_ms=self.timeout_ms)
            job = self.configured_model.run_async([self.bindings], _callback)
            job.wait(self.timeout_ms)
            completion_info = completion_holder.get("info")
            if completion_info is None:
                raise RuntimeError("Hailo async job completed without callback information.")
            exception = getattr(completion_info, "exception", None)
            if exception:
                raise RuntimeError(f"Hailo async inference exception: {exception}")
        else:
            self.configured_model.run([self.bindings], timeout=self.timeout_ms)
        raw = self.bindings.output().get_buffer()
        return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

    def _run_single(self, image_bgr: np.ndarray) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        scale = min(MODEL_INPUT_SIZE / width, MODEL_INPUT_SIZE / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(
            image_bgr,
            (new_width, new_height),
            interpolation=cv2.INTER_LINEAR,
        )
        top = (MODEL_INPUT_SIZE - new_height) // 2
        left = (MODEL_INPUT_SIZE - new_width) // 2
        canvas = np.zeros(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3),
            dtype=np.uint8,
        )
        canvas[top:top + new_height, left:left + new_width] = resized
        output = self._infer_tile_rgb(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        crop_top = int(top * MODEL_SCALE)
        crop_left = int(left * MODEL_SCALE)
        crop = output[
            crop_top:crop_top + new_height * MODEL_SCALE,
            crop_left:crop_left + new_width * MODEL_SCALE,
        ]
        return cv2.resize(
            crop,
            (width * MODEL_SCALE, height * MODEL_SCALE),
            interpolation=cv2.INTER_LINEAR,
        )

    def _run_tiled(
        self,
        image_bgr: np.ndarray,
        on_tile: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        cols, rows = tile_grid(width, height)
        total = cols * rows
        output = np.zeros(
            (height * MODEL_SCALE, width * MODEL_SCALE, 3),
            dtype=np.uint8,
        )
        index = 0

        for row in range(rows):
            for col in range(cols):
                index += 1
                if on_tile:
                    on_tile(index, total)

                tile_x = col * TILE_STRIDE
                tile_y = row * TILE_STRIDE
                tile_w = min(TILE_STRIDE, width - tile_x)
                tile_h = min(TILE_STRIDE, height - tile_y)

                context_left = min(TILE_PAD, tile_x)
                context_top = min(TILE_PAD, tile_y)
                context_right = min(TILE_PAD, max(0, width - (tile_x + tile_w)))
                context_bottom = min(TILE_PAD, max(0, height - (tile_y + tile_h)))

                x1 = tile_x - context_left
                y1 = tile_y - context_top
                x2 = tile_x + tile_w + context_right
                y2 = tile_y + tile_h + context_bottom
                patch = image_bgr[y1:y2, x1:x2]

                pad_top = TILE_PAD - context_top
                pad_left = TILE_PAD - context_left
                pad_bottom = MODEL_INPUT_SIZE - patch.shape[0] - pad_top
                pad_right = MODEL_INPUT_SIZE - patch.shape[1] - pad_left
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    patch = cv2.copyMakeBorder(
                        patch,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_REFLECT_101,
                    )

                input_extract_x = pad_left + context_left
                input_extract_y = pad_top + context_top
                output_tile = self._infer_tile_rgb(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                output_x = input_extract_x * MODEL_SCALE
                output_y = input_extract_y * MODEL_SCALE
                output[
                    tile_y * MODEL_SCALE:tile_y * MODEL_SCALE + tile_h * MODEL_SCALE,
                    tile_x * MODEL_SCALE:tile_x * MODEL_SCALE + tile_w * MODEL_SCALE,
                ] = output_tile[
                    output_y:output_y + tile_h * MODEL_SCALE,
                    output_x:output_x + tile_w * MODEL_SCALE,
                ]
        return output
