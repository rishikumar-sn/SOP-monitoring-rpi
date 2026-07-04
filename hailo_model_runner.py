import os
import re
import threading
import logging
import time
from importlib import metadata
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

try:
    from hailo_platform import HEF, VDevice, FormatType
    try:
        from hailo_platform import HailoSchedulingAlgorithm
    except ImportError:
        HailoSchedulingAlgorithm = None
    HAILO_AVAILABLE = True
except ImportError:
    HEF = None
    VDevice = None
    FormatType = None
    HailoSchedulingAlgorithm = None
    HAILO_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_HAILO_INFERENCE_TIMEOUT_MS = 30000
DEFAULT_HAILO_BATCH_SIZE = 1
SLOW_HAILO_INFERENCE_MS = 1000.0
HAILORT_DFC_COMPATIBILITY = {
    (4, 20): (3, 30),
    (4, 21): (3, 31),
    (4, 22): (3, 32),
    (4, 23): (3, 33),
}


def _parse_version(value: str) -> Tuple[int, ...]:
    match = re.search(r"\d+(?:\.\d+)+", str(value or ""))
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(0).split("."))


def _read_hef_compiler_version(hef_path: str) -> str:
    try:
        with open(hef_path, "rb") as hef_file:
            header = hef_file.read(512).decode("ascii", errors="ignore")
    except OSError:
        return "unknown"

    match = re.search(r"\d+\.\d+\.\d+", header)
    return match.group(0) if match else "unknown"


def _validate_hef_runtime_compatibility(
    hef_path: str,
    model_name: str,
    hailort_version: str,
) -> str:
    compiler_version = _read_hef_compiler_version(hef_path)
    runtime = _parse_version(hailort_version)
    compiler = _parse_version(compiler_version)
    supported_dfc = HAILORT_DFC_COMPATIBILITY.get(runtime[:2])

    if supported_dfc is not None and compiler[:2] > supported_dfc:
        logger.warning(
            "[%s] HEF compiler/runtime mismatch for %s: "
            "header DFC=%s, HailoRT=%s (usual DFC %s.%s.x). "
            "The HEF may load and configure but still fail when its first "
            "inference is submitted. Recompile with the paired DFC release or "
            "upgrade the complete HailoRT driver, firmware, library, and Python stack.",
            model_name,
            os.path.basename(hef_path),
            compiler_version,
            hailort_version,
            supported_dfc[0],
            supported_dfc[1],
        )
        return compiler_version

    if supported_dfc is None or not compiler:
        logger.warning(
            "[%s] Could not verify HEF/runtime release compatibility for %s: "
            "header compiler=%s, HailoRT=%s.",
            model_name,
            os.path.basename(hef_path),
            compiler_version,
            hailort_version,
        )
        return compiler_version

    logger.info(
        "[%s] HEF/runtime release check passed for %s: DFC=%s, HailoRT=%s.",
        model_name,
        os.path.basename(hef_path),
        compiler_version,
        hailort_version,
    )
    return compiler_version


class HailoHEFModel:
    """Robust wrapper around a single Hailo HEF model for synchronous-like inference using the modern async API."""

    def __init__(
        self,
        vdevice: Any,
        hef_path: str,
        name: str,
        inference_lock: threading.Lock,
        hailort_version: str,
        timeout_ms: int | None = None,
        batch_size: int = DEFAULT_HAILO_BATCH_SIZE,
    ):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF not found for {name}: {hef_path}")

        self.name = name
        self.hef_path = hef_path
        self.hailort_version = hailort_version
        self.hef_compiler_version = _validate_hef_runtime_compatibility(
            hef_path,
            name,
            hailort_version,
        )
        self._inference_lock = inference_lock
        configured_timeout = os.environ.get("HAILO_INFERENCE_TIMEOUT_MS")
        self.timeout_ms = max(
            10000,
            int(
                configured_timeout
                if configured_timeout is not None
                else (
                    timeout_ms
                    if timeout_ms is not None
                    else DEFAULT_HAILO_INFERENCE_TIMEOUT_MS
                )
            ),
        )
        self.hef = HEF(hef_path)
        requested_batch_size = max(1, int(batch_size))
        if requested_batch_size != 1:
            logger.warning(
                "[%s] Ignoring requested runtime batch size %s; "
                "the application submits one frame per inference.",
                self.name,
                requested_batch_size,
            )
        self.batch_size = 1
        self.last_inference_ms = 0.0

        self.infer_model = vdevice.create_infer_model(hef_path)
        self.infer_model.set_batch_size(self.batch_size)

        self.input_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = self.input_info.name
        self.input_shape = tuple(self.input_info.shape)
        
        # Infer input layout
        if len(self.input_shape) == 3:
            if self.input_shape[2] in (1, 3, 4):
                self.input_layout = "HWC"
                self.input_h, self.input_w, self.input_c = self.input_shape
            elif self.input_shape[0] in (1, 3, 4):
                self.input_layout = "CHW"
                self.input_c, self.input_h, self.input_w = self.input_shape
            else:
                raise RuntimeError(f"[{self.name}] Unsupported input shape: {self.input_shape}")
        else:
             raise RuntimeError(f"[{self.name}] Unsupported input rank: {len(self.input_shape)}")

        self.native_input_dtype = self._format_type_to_numpy(
            self.input_info.format.type
        )
        requested_input_format = os.environ.get(
            "HAILO_INPUT_FORMAT",
            "native",
        ).strip().lower()
        if requested_input_format not in ("native", "uint8", "float32"):
            raise ValueError(
                "HAILO_INPUT_FORMAT must be native, uint8, or float32; "
                f"received {requested_input_format!r}."
            )
        self.input_dtype = self.native_input_dtype

        # Allow an explicit host format for diagnosing platform-specific DMA
        # behavior while preserving the HEF-native format by default.
        if requested_input_format == "float32":
            try:
                self.infer_model.input().set_format_type(FormatType.FLOAT32)
                self.input_dtype = np.float32
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.name}] Could not configure FLOAT32 input buffers: {exc}"
                ) from exc
        elif requested_input_format == "uint8":
            try:
                self.infer_model.input().set_format_type(FormatType.UINT8)
                self.input_dtype = np.uint8
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.name}] Could not configure UINT8 input buffers: {exc}"
                ) from exc
        elif self.input_dtype != np.float32:
            try:
                self.infer_model.input().set_format_type(FormatType.UINT8)
                self.input_dtype = np.uint8
            except Exception:
                pass
        else:
            try:
                self.infer_model.input().set_format_type(FormatType.FLOAT32)
                self.input_dtype = np.float32
            except Exception:
                pass

        self.output_infos = list(self.hef.get_output_vstream_infos())
        self.output_names = [info.name for info in self.output_infos]
        self.output_dtypes: Dict[str, np.dtype] = {}

        for info in self.output_infos:
            name_ = info.name
            dtype = self._format_type_to_numpy(info.format.type)
            try:
                self.infer_model.output(name_).set_format_type(FormatType.FLOAT32)
                dtype = np.float32
            except Exception:
                pass
            self.output_dtypes[name_] = dtype

        self._config_ctx = self.infer_model.configure()
        self.configured_model = self._config_ctx.__enter__()
        self._logged_input_buffer = False

        logger.info(
            "[%s] HEF loaded: %s | runtime_batch=%s | "
            "input=%s(%s,%s,%s) | native_dtype=%s host_dtype=%s | outputs=%s",
            self.name,
            os.path.basename(self.hef_path),
            self.batch_size,
            self.input_layout,
            self.input_h,
            self.input_w,
            self.input_c,
            np.dtype(self.native_input_dtype),
            np.dtype(self.input_dtype),
            self.output_names,
        )

    def _prepare_input_buffer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Convert caller input to the exact unbatched HEF shape and dtype."""
        tensor = np.asarray(input_tensor)
        if tensor.size == 0:
            raise ValueError(f"[{self.name}] Input tensor is empty.")

        if tensor.ndim == 4:
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"[{self.name}] Only batch size 1 is supported; received {tensor.shape}."
                )
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(
                f"[{self.name}] Expected a 3D image tensor or batch-1 tensor; received {tensor.shape}."
            )

        expected_shape = tuple(int(value) for value in self.input_shape)
        if tensor.shape != expected_shape:
            if self.input_layout == "HWC":
                chw_shape = (self.input_c, self.input_h, self.input_w)
                if tensor.shape == chw_shape:
                    tensor = np.transpose(tensor, (1, 2, 0))
            else:
                hwc_shape = (self.input_h, self.input_w, self.input_c)
                if tensor.shape == hwc_shape:
                    tensor = np.transpose(tensor, (2, 0, 1))

        if tensor.shape != expected_shape:
            raise ValueError(
                f"[{self.name}] Input shape {tensor.shape} does not match HEF "
                f"{self.input_layout} shape {expected_shape}."
            )

        if tensor.dtype != self.input_dtype:
            if self.input_dtype == np.uint8 and np.issubdtype(tensor.dtype, np.floating):
                finite_max = float(np.nanmax(tensor))
                if finite_max <= 1.0:
                    tensor = tensor * 255.0
                tensor = np.clip(tensor, 0, 255).astype(np.uint8)
            else:
                tensor = tensor.astype(self.input_dtype)

        # Hailo's Python bindings retain a direct reference to this array.
        # Use an owned C-order allocation rather than a view from transpose.
        tensor = np.array(tensor, dtype=self.input_dtype, order="C", copy=True)
        expected_bytes = int(np.prod(expected_shape)) * np.dtype(self.input_dtype).itemsize
        if tensor.nbytes != expected_bytes:
            raise ValueError(
                f"[{self.name}] Prepared input has {tensor.nbytes} bytes; "
                f"HEF expects {expected_bytes} bytes for {expected_shape} {self.input_dtype}."
            )

        if not self._logged_input_buffer:
            self._logged_input_buffer = True
            logger.info(
                "[%s] Prepared input buffer: shape=%s dtype=%s bytes=%s",
                self.name,
                tensor.shape,
                tensor.dtype,
                tensor.nbytes,
            )
        return tensor

    def _format_type_to_numpy(self, format_type: Any) -> np.dtype:
        name = str(format_type).split(".")[-1].upper()
        mapping = {
            "FLOAT32": np.float32,
            "UINT8": np.uint8,
            "UINT16": np.uint16,
            "INT8": np.int8,
            "INT16": np.int16,
        }
        return mapping.get(name, np.float32)

    def _create_output_buffers(self) -> Dict[str, np.ndarray]:
        buffers: Dict[str, np.ndarray] = {}
        for name_ in self.output_names:
            # Note: infer_model.output(name_).shape is usually HWC even if HEF was compiled CHW, 
            # as HailoRT handles the layout conversion if set_format_type is used.
            shape = tuple(self.infer_model.output(name_).shape)
            dtype = self.output_dtypes.get(name_, np.float32)
            buffers[name_] = np.empty(shape, dtype=dtype)
        return buffers

    def run_inference(
        self,
        input_tensor: np.ndarray,
        cancel_event: threading.Event | None = None,
    ) -> Any:
        """Run synchronous inference by waiting on async callback.

        Keep the shared runtime lock until the async job completes. Multiple
        configured models share one physical Hailo device, so allowing another
        model to submit while this job is active can abort streams.
        """
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError(f"[{self.name}] Inference cancelled before submission")

        input_tensor = self._prepare_input_buffer(input_tensor)

        output_buffers = self._create_output_buffers()
        binding = self.configured_model.create_bindings(
            input_buffers={self.input_name: input_tensor},
            output_buffers=output_buffers,
        )
        completion_holder: Dict[str, Any] = {"info": None}

        def _callback(completion_info: Any) -> None:
            completion_holder["info"] = completion_info

        started_at = time.perf_counter()
        try:
            with self._inference_lock:
                self.configured_model.wait_for_async_ready(timeout_ms=self.timeout_ms)
                job = self.configured_model.run_async([binding], _callback)
                # HailoRT wait() also waits for the completion callback.
                job.wait(self.timeout_ms)
        except Exception as exc:
            self.last_inference_ms = (time.perf_counter() - started_at) * 1000.0
            normalized_error = str(exc).lower()
            if "timeout" in normalized_error or "stream was aborted" in normalized_error:
                guidance = (
                    "The configured Hailo async pipeline did not complete this job. "
                    "Do not retry this configured model in place; check preceding "
                    "host_error/device_error messages and whether another resident "
                    "network was active during the model switch."
                )
            else:
                guidance = (
                    "Check the preceding HailoRT log for the first host, device, "
                    "driver, or model configuration error."
                )
            raise RuntimeError(
                f"[{self.name}] Hailo inference failed after binding "
                f"{self.input_name}: shape={input_tensor.shape}, "
                f"dtype={input_tensor.dtype}, bytes={input_tensor.nbytes}, "
                f"NumPy={np.__version__}, HailoRT={self.hailort_version}, "
                f"HEF compiler={self.hef_compiler_version}, "
                f"runtime_batch={self.batch_size}, elapsed={self.last_inference_ms:.1f}ms, "
                f"cause={exc}. "
                f"{guidance}"
            ) from exc
        self.last_inference_ms = (time.perf_counter() - started_at) * 1000.0
        if self.last_inference_ms >= SLOW_HAILO_INFERENCE_MS:
            logger.warning(
                "[%s] Slow Hailo inference: %.1fms (runtime_batch=%s)",
                self.name,
                self.last_inference_ms,
                self.batch_size,
            )

        completion_info = completion_holder.get("info")
        if completion_info is None:
            raise RuntimeError(
                f"[{self.name}] Hailo job completed without callback information."
            )
        if completion_info is not None and getattr(completion_info, "exception", None):
            raise RuntimeError(f"[{self.name}] Inference exception: {completion_info.exception}")

        if len(self.output_names) == 1:
            return binding.output().get_buffer()

        return {name_: binding.output(name_).get_buffer() for name_ in self.output_names}

    def validate_runtime_contract(self, probe_runs: int = 1) -> Dict[str, Any]:
        """Submit real zero-buffer jobs to prove this configured HEF can run."""
        run_count = max(1, int(probe_runs))
        probe = np.zeros(self.input_shape, dtype=self.input_dtype)
        output_schema: Any = None
        for run_index in range(run_count):
            try:
                output = self.run_inference(probe)
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.name}] Startup Hailo self-test failed on "
                    f"probe {run_index + 1}: {exc}"
                ) from exc

            if isinstance(output, dict):
                missing_outputs = [
                    name for name in self.output_names if name not in output
                ]
                if missing_outputs:
                    raise RuntimeError(
                        f"[{self.name}] Startup Hailo self-test missed output(s): "
                        + ", ".join(missing_outputs)
                    )
                output_schema = {
                    name: tuple(np.asarray(value).shape)
                    for name, value in output.items()
                }
            else:
                tensor = np.asarray(output)
                if tensor.size == 0:
                    raise RuntimeError(
                        f"[{self.name}] Startup Hailo self-test returned an empty output."
                    )
                output_schema = tuple(tensor.shape)

        return {
            "name": self.name,
            "probe_runs": run_count,
            "last_inference_ms": float(self.last_inference_ms),
            "output_schema": output_schema,
        }

    def close(self) -> None:
        if self._config_ctx is not None:
            try:
                self._config_ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._config_ctx = None
        self.configured_model = None
        self.infer_model = None
        self.hef = None

class HailoRuntime:
    """Shared VDevice context for multiple HEF models on Hailo-8."""

    def __init__(self):
        self.last_model_error = ""
        if not HAILO_AVAILABLE:
            self.vdevice = None
            self.last_model_error = "hailo_platform is not installed or could not be imported."
            return

        numpy_version = np.__version__
        try:
            numpy_major = int(numpy_version.split(".", 1)[0])
        except (TypeError, ValueError):
            numpy_major = 0
        hailort_version = "unknown"
        for distribution_name in ("hailort", "hailo-platform", "hailo_platform"):
            try:
                hailort_version = metadata.version(distribution_name)
                break
            except metadata.PackageNotFoundError:
                continue
        logger.info(
            "Hailo Python environment: NumPy=%s HailoRT=%s binding=%s",
            numpy_version,
            hailort_version,
            getattr(HEF, "__module__", "hailo_platform"),
        )
        print(
            f"[HailoRuntime] Python environment: NumPy={numpy_version}, "
            f"HailoRT={hailort_version}"
        )
        if numpy_major >= 2:
            raise RuntimeError(
                "HailoRT Python input bindings are incompatible with NumPy "
                f"{numpy_version} on this Raspberry Pi setup and can report input buffer size 0. "
                "Install NumPy 1.26.4 in the Python environment that starts this app."
            )

        self.hailort_version = hailort_version
        self.inference_lock = threading.Lock()
        self.vdevice = self._create_vdevice()
        self.models: List[HailoHEFModel] = []

    def _create_vdevice(self) -> Any:
        try:
            params = VDevice.create_params()
            if HailoSchedulingAlgorithm is not None and hasattr(params, "scheduling_algorithm"):
                params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            if hasattr(params, "group_id"):
                params.group_id = "JEWELRY_APP_SHARED"
            return VDevice(params)
        except Exception as e:
            logger.warning("Could not create shared VDevice params (%s). Falling back to default VDevice().", e)
            return VDevice()

    def create_model(
        self,
        hef_path: str,
        name: str,
        timeout_ms: int | None = None,
        batch_size: int = DEFAULT_HAILO_BATCH_SIZE,
    ) -> Optional[HailoHEFModel]:
        if not HAILO_AVAILABLE:
            self.last_model_error = "hailo_platform is not installed or could not be imported."
            return None
        if not os.path.exists(hef_path):
            self.last_model_error = f"HEF file is missing for {name}: {hef_path}"
            return None
        try:
            model = HailoHEFModel(
                self.vdevice,
                hef_path,
                name,
                self.inference_lock,
                self.hailort_version,
                timeout_ms=timeout_ms,
                batch_size=batch_size,
            )
            self.models.append(model)
            self.last_model_error = ""
            return model
        except Exception as e:
            detail = str(e)
            if (
                "driver ioctl" in detail.lower()
                or "hailo_driver_operation_failed" in detail.lower()
                or "fw_control" in detail.lower()
            ):
                detail += (
                    " The kernel could not communicate with the Hailo device. "
                    "On the Raspberry Pi run `dmesg -T | grep -iE "
                    "'hailo|pcie|aer' | tail -100` and `hailortcli scan`. "
                    "If standalone `hailortcli run MODEL.hef` works, suspect a stale "
                    "VDevice/model handoff or too many simultaneously configured "
                    "models in the application rather than a corrupt HEF."
                )
            self.last_model_error = f"{name}: {detail}"
            logger.error("Failed to create Hailo model %s: %s", name, detail)
            return None

    def close(self) -> None:
        for model in self.models:
            model.close()
        self.models.clear()

        if self.vdevice is not None and hasattr(self.vdevice, "release"):
            try:
                self.vdevice.release()
            except Exception:
                pass
        self.vdevice = None
