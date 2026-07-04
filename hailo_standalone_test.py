from __future__ import annotations

import argparse
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from hailo_model_runner import HAILO_AVAILABLE, HailoRuntime


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = {
    "stone": BASE_DIR / "models" / "yolov8s_seg.hef",
    "gold": BASE_DIR / "models" / "gold.hef",
    "acid": BASE_DIR / "models" / "bestnewacid.hef",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Hailo inference without the camera, GUI, audio, or purity manager."
        )
    )
    parser.add_argument(
        "--scenario",
        choices=("cli", "single", "resident", "each"),
        default="single",
        help=(
            "cli: test the selected HEF using hailortcli only; "
            "single: load only the selected model; "
            "resident: load all three models, then test the selected model; "
            "each: test every model in a fresh runtime."
        ),
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PATHS),
        default="stone",
        help="Model to exercise in single or resident mode.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3000,
        help="Number of inference submissions per tested model.",
    )
    parser.add_argument(
        "--input-format",
        choices=("native", "uint8", "float32"),
        default="native",
        help=(
            "Host input buffer format for Python scenarios. "
            "Use float32 to reproduce the working Pi path."
        ),
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        help="Skip the optional hailortcli scan preflight.",
    )
    return parser.parse_args()


def package_version(*names: str) -> str:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def print_environment() -> None:
    print("=== Environment ===")
    print(f"host={platform.node() or 'unknown'}")
    print(f"platform={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"numpy={np.__version__}")
    print(
        "hailort="
        + package_version("hailort", "hailo-platform", "hailo_platform")
    )
    print()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_model_identity(model_name: str) -> None:
    model_path = MODEL_PATHS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_name} HEF: {model_path}")
    print(f"model_path={model_path}")
    print(f"model_size={model_path.stat().st_size}")
    print(f"model_sha256={sha256_file(model_path)}")
    print()


def run_cli_command(
    title: str,
    command: list[str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str] | None:
    print(f"=== {title} ===")
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "").strip()
        error = (exc.stderr or "").strip()
        if output:
            print(output)
        if error:
            print(error)
        print(f"{title}: TIMEOUT after {timeout_seconds}s")
        print()
        return None
    except Exception as exc:
        print(f"{title}: ERROR ({exc})")
        print()
        return None

    output = (result.stdout or "").strip()
    error = (result.stderr or "").strip()
    if output:
        print(output)
    if error:
        print(error)
    print(f"{title} exit_code={result.returncode}")
    print()
    return result


def run_hailort_scan() -> None:
    executable = shutil.which("hailortcli")
    if executable is None:
        print("hailortcli scan: SKIPPED (hailortcli not found)")
        print()
        return

    run_cli_command(
        "hailortcli scan",
        [executable, "scan"],
        timeout_seconds=15,
    )


def run_hailort_cli_model_test(model_name: str) -> None:
    executable = shutil.which("hailortcli")
    if executable is None:
        raise RuntimeError("hailortcli was not found in PATH")

    model_path = MODEL_PATHS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_name} HEF: {model_path}")

    identify_result = run_cli_command(
        "hailortcli fw-control identify",
        [executable, "fw-control", "identify"],
        timeout_seconds=15,
    )
    parse_result = run_cli_command(
        f"hailortcli parse-hef ({model_name})",
        [executable, "parse-hef", str(model_path)],
        timeout_seconds=15,
    )
    run_result = run_cli_command(
        f"hailortcli run one frame ({model_name})",
        [
            executable,
            "run",
            str(model_path),
            "--frames-count",
            "1",
        ],
        timeout_seconds=60,
    )

    if identify_result is None or identify_result.returncode != 0:
        raise RuntimeError("hailortcli could not identify the Hailo device")
    if parse_result is None or parse_result.returncode != 0:
        raise RuntimeError(f"hailortcli could not parse {model_path.name}")
    if run_result is None:
        raise RuntimeError(
            f"hailortcli could not complete one frame for {model_path.name}; "
            "the failure is below the Python application layer"
        )
    if run_result.returncode != 0:
        raise RuntimeError(
            f"hailortcli run failed for {model_path.name} "
            f"with exit code {run_result.returncode}"
        )


def make_probe(model: Any, seed: int) -> np.ndarray:
    shape = tuple(int(value) for value in model.input_shape)
    dtype = np.dtype(model.input_dtype)
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(0, 256, size=shape, dtype=dtype)
    return rng.random(shape, dtype=np.float32).astype(dtype, copy=False)


def validate_outputs(model: Any, output: Any) -> str:
    if isinstance(output, dict):
        missing = sorted(set(model.output_names).difference(output))
        if missing:
            raise RuntimeError("missing output channels: " + ", ".join(missing))
        empty = [
            name for name, value in output.items() if np.asarray(value).size == 0
        ]
        if empty:
            raise RuntimeError("empty output channels: " + ", ".join(empty))
        return ", ".join(
            f"{name}:{tuple(np.asarray(value).shape)}"
            for name, value in output.items()
        )

    tensor = np.asarray(output)
    if tensor.size == 0:
        raise RuntimeError("model returned an empty output tensor")
    return str(tuple(tensor.shape))


def create_model(runtime: HailoRuntime, model_name: str) -> Any:
    model_path = MODEL_PATHS[model_name]
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {model_name} HEF: {model_path}")

    model = runtime.create_model(str(model_path), model_name.upper())
    if model is None:
        raise RuntimeError(
            runtime.last_model_error or f"Could not create {model_name} model"
        )
    return model


def exercise_model(model: Any, runs: int) -> None:
    run_count = max(1, int(runs))
    timings: list[float] = []
    output_schema = ""

    print(
        f"Testing {model.name}: input={tuple(model.input_shape)} "
        f"dtype={np.dtype(model.input_dtype)} runs={run_count}"
    )
    for run_index in range(run_count):
        probe = make_probe(model, seed=run_index)
        started_at = time.perf_counter()
        output = model.run_inference(probe)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        output_schema = validate_outputs(model, output)
        timings.append(elapsed_ms)
        print(
            f"[{model.name}] run {run_index + 1:02d}/{run_count:02d} "
            f"PASS {elapsed_ms:.1f} ms"
        )

    print(
        f"[{model.name}] RESULT=PASS "
        f"min={min(timings):.1f} ms "
        f"avg={sum(timings) / len(timings):.1f} ms "
        f"max={max(timings):.1f} ms"
    )
    print(f"[{model.name}] outputs={output_schema}")


def run_single(model_name: str, runs: int) -> None:
    runtime = HailoRuntime()
    try:
        model = create_model(runtime, model_name)
        exercise_model(model, runs)
    finally:
        runtime.close()


def run_resident(model_name: str, runs: int) -> None:
    runtime = HailoRuntime()
    try:
        models = {
            name: create_model(runtime, name)
            for name in ("stone", "gold", "acid")
        }
        print("All three HEFs are configured in one shared VDevice.")
        exercise_model(models[model_name], runs)
    finally:
        runtime.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print_environment()
    print_model_identity(args.model)
    if not args.skip_scan:
        run_hailort_scan()

    if not HAILO_AVAILABLE:
        print("RESULT=FAIL: hailo_platform could not be imported.", file=sys.stderr)
        return 2

    os.environ["HAILO_INPUT_FORMAT"] = args.input_format
    if args.scenario != "cli":
        print(f"python_input_format={args.input_format}")
        print()

    try:
        if args.scenario == "cli":
            run_hailort_cli_model_test(args.model)
        elif args.scenario == "single":
            run_single(args.model, args.runs)
        elif args.scenario == "resident":
            run_resident(args.model, args.runs)
        else:
            for model_name in ("stone", "gold", "acid"):
                print(f"=== Fresh runtime: {model_name.upper()} ===")
                run_single(model_name, args.runs)
                print()
    except Exception as exc:
        logging.exception("Standalone Hailo test failed")
        print(f"RESULT=FAIL: {exc}", file=sys.stderr)
        return 1

    print("RESULT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
