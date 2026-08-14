from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


class OnnxObbModel:
    """Long-lived ONNX Runtime session for the exported LCD OBB model."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"LCD ONNX model is missing: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(
                f"Expected one ONNX input/output, received {len(inputs)}/{len(outputs)}"
            )
        self.input_name = inputs[0].name
        self.input_shape = tuple(inputs[0].shape)
        self.output_names = [outputs[0].name]
        self.output_shape = tuple(outputs[0].shape)
        self.configuration_count = 1
        self.inference_count = 0
        self.last_inference_ms = 0.0

    def contract(self):
        return {
            "backend": "ONNX Runtime CPU",
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "input_dtype": "float32",
            "output_names": list(self.output_names),
            "output_shape": self.output_shape,
            "configuration_count": self.configuration_count,
        }

    def infer(self, model_rgb):
        tensor = np.asarray(model_rgb)
        if tensor.shape != (512, 512, 3):
            raise ValueError(f"Unexpected ONNX input image shape: {tensor.shape}")
        if tensor.dtype != np.uint8:
            raise ValueError(f"ONNX input image must be uint8, received {tensor.dtype}")
        input_tensor = np.ascontiguousarray(
            np.transpose(tensor.astype(np.float32) / 255.0, (2, 0, 1))[None]
        )

        started = time.perf_counter()
        values = self.session.run(self.output_names, {self.input_name: input_tensor})
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0
        self.inference_count += 1
        return dict(zip(self.output_names, values))
