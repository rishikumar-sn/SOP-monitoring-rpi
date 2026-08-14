# LCD Weight Reader

The PyQt6 app captures one full-resolution camera frame, crops the user-drawn
white-board ROI, detects one full LCD with the OBB model and rotated NMS,
expands the detected quad to protect edge digits, and perspective-warps it to a
straight LCD image.

Digit recognition uses the project-local English PP-OCRv5 mobile recognition
model. Two slightly different crops of the same rectified LCD are recognized;
the app accepts a reading only when both crops return identical digits and the
lower confidence is at least 0.70. It otherwise shows `READ FAILED` instead of
returning a questionable weight. A successful digit string is displayed with
two hardcoded decimal places by the downstream weight formatting flow.

ONNX Runtime and PaddlePaddle crash when their native libraries are initialized
in the same Raspberry Pi process. The GUI therefore keeps the detector in its
worker and starts one persistent project-local PaddleOCR subprocess. The OCR
model loads once when the app starts; this is still a single operator app and a
single-frame capture flow.

The detector is the long-lived ONNX Runtime CPU model in
`models/lcd_obb.onnx`. No LCD HEF or Hailo device is used by this project.

## Validation result

On the current 55 labeled saved captures, each individual Paddle crop reads 52
correctly and 3 incorrectly. Requiring the two crops to agree at confidence
0.70 changes this to **50 correct, 0 wrong, and 5 explicit failures**. This is
the reason the app does not accept a single Paddle prediction directly.

The previous fixed-slot seven-segment decoder and five-frame burst consensus
have been removed from production. The labeled-input preparation and evaluator
remain under `experiments/` so the same acceptance check can be repeated after
collecting more samples.

## Run

From this directory:

```bash
python3 main.py
```

PaddleOCR is installed in `.venv-paddleocr`, and its model is stored under
`models/paddleocr/`. The isolated environment is intentional: the ONNX detector
and PaddlePaddle native runtimes crash when loaded into the same Raspberry Pi
process. `ocr/paddle_client.py` launches the environment by absolute path, so a
parent application can use it from any working directory. To recreate it:

```bash
python3 -m venv --system-site-packages .venv-paddleocr
.venv-paddleocr/bin/pip install -r requirements-paddleocr.txt
```

## Save validation samples

After `Capture & Read`, enter the true visible LCD value, select the condition,
and click `Save Validation Sample`. Each sample is saved under
`validation/phase8/samples/` with its clean ROI, full frame, detector and warp
evidence, both Paddle crops, and `metadata.json`. Failed reads and LCD-not-found
captures can also be saved.

## Repeat checks

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tests/onnx_reference_smoke.py
python3 tests/phase5_acceptance.py
python3 tests/phase6_acceptance.py
python3 tests/phase7_acceptance.py
python3 experiments/prepare_paddleocr_inputs.py
.venv-paddleocr/bin/python experiments/paddleocr_eval.py
python3 tests/phase8_acceptance.py
QT_QPA_PLATFORM=offscreen python3 tests/phase5_gui_smoke.py
```
