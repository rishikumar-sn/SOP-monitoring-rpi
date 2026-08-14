"""Client for the isolated project-local PaddleOCR process."""

import json
from pathlib import Path
import selectors
import subprocess


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PaddleOCRClient:
    def __init__(self, startup_timeout=30.0):
        python_path = PROJECT_DIR / ".venv-paddleocr" / "bin" / "python"
        service_path = PROJECT_DIR / "ocr" / "paddle_service.py"
        if not python_path.is_file():
            raise FileNotFoundError(f"PaddleOCR environment is missing: {python_path}")

        log_path = PROJECT_DIR / "logs" / "paddleocr_service.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [str(python_path), "-u", str(service_path)],
            cwd=str(PROJECT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log_file,
            text=True,
            bufsize=1,
        )
        try:
            message = self._read_message(startup_timeout)
        except Exception:
            self.close()
            raise
        if message.get("status") != "ready":
            self.close()
            raise RuntimeError(f"PaddleOCR service failed to start: {message}")

    def _read_message(self, timeout):
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            while True:
                if not selector.select(timeout):
                    raise TimeoutError("Timed out waiting for PaddleOCR service")
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError(
                        f"PaddleOCR service exited with code {self._process.poll()}"
                    )
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        finally:
            selector.close()

    def recognize(self, paths, timeout=15.0):
        if self._process.poll() is not None:
            raise RuntimeError("PaddleOCR service is not running")
        request = {"op": "recognize", "paths": [str(path) for path in paths]}
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        result = self._read_message(timeout)
        if result.get("status") != "ok":
            raise RuntimeError(result.get("message", "PaddleOCR inference failed"))
        return result

    def close(self):
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.write('{"op":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                process.wait(timeout=5)
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
            self._log_file = None
