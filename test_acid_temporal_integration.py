from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from acid_temporal_inference import TemporalAcidClassifier, prepare_temporal_triplet
from purity_test_manager import PurityTestManager


class _FakeOnnxSession:
    def __init__(self, logits: np.ndarray):
        self.logits = logits

    def run(self, _outputs, _inputs):
        return [self.logits]


class _FakeTemporalModel:
    def __init__(self, prediction: str):
        self.prediction = prediction

    def predict(self, _baseline, _transition, _observation):
        return {
            "prediction": self.prediction,
            "confidence": 0.99,
            "probabilities": {
                "gold_22k": 0.99 if self.prediction == "gold_22k" else 0.005,
                "non_gold": 0.99 if self.prediction == "non_gold" else 0.005,
                "invalid": 0.99 if self.prediction == "invalid" else 0.005,
            },
            "latency_ms": 10.0,
        }


class AcidTemporalInferenceTests(unittest.TestCase):
    def test_deployed_model_contract_and_identity(self):
        model_path = Path(__file__).resolve().parent / "models" / "acid_temporal_mobilenet_v3.onnx"
        classifier = TemporalAcidClassifier(model_path)

        self.assertEqual(
            "bac996af8b5ad049b63571c5ae96ea588ae1c719884e7662deaf14b066b448bd",
            classifier.sha256,
        )

    def test_triplet_contract(self):
        baseline = [np.full((30, 60, 3), value, np.uint8) for value in (10, 20)]
        transition = [np.full((30, 60, 3), value, np.uint8) for value in (30, 40, 50)]
        observation = [np.full((30, 60, 3), value, np.uint8) for value in range(60, 120, 10)]

        triplet = prepare_temporal_triplet(baseline, transition, observation)

        self.assertEqual((3, 224, 224, 3), triplet.shape)
        self.assertEqual(np.uint8, triplet.dtype)

    def test_low_confidence_gold_becomes_invalid(self):
        classifier = TemporalAcidClassifier.__new__(TemporalAcidClassifier)
        classifier.session = _FakeOnnxSession(
            np.array([[1.5, 0.1, 0.0]], dtype=np.float32)
        )
        frame = np.zeros((20, 20, 3), dtype=np.uint8)

        result = classifier.predict([frame], [frame], [frame])

        self.assertEqual("invalid", result["prediction"])


class PurityTemporalManagerTests(unittest.TestCase):
    def _manager(self, base_dir: Path, spoken: list[str] | None = None):
        return PurityTestManager(
            base_dir=base_dir,
            frame_getter=lambda: np.zeros((80, 100, 3), dtype=np.uint8),
            speak_fn=(spoken if spoken is not None else []).append,
        )

    def test_start_acid_observation_requires_ready_stage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = self._manager(Path(temporary_dir))
            manager._module = SimpleNamespace(STATE={"stage": "READY_FOR_ACID"})
            manager._temporal_model = _FakeTemporalModel("gold_22k")
            manager._session.running = True
            manager._session.stage = "READY_FOR_ACID"

            state = manager.start_acid_observation()

            self.assertEqual("BASELINE_2S", state["stage"])
            self.assertFalse(state["acid_result_ready"])

    def test_temporal_result_keeps_only_report_images_and_releases_buffers(self):
        spoken: list[str] = []
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manager = self._manager(root, spoken)
            manager._module = SimpleNamespace(
                STATE={"stage": "OBSERVE_5S", "acid_positive_streak": 3}
            )
            manager._temporal_model = _FakeTemporalModel("gold_22k")
            manager._session_root = root / "purity"
            manager._session.started_at = "started"
            manager._session.running = True
            frame = np.zeros((40, 50, 3), dtype=np.uint8)
            manager._baseline_crops = [frame.copy() for _ in range(5)]
            manager._transition_crops = [frame.copy() for _ in range(3)]
            manager._observation_crops = [frame.copy() for _ in range(10)]
            manager._reaction_bbox = (5, 5, 30, 30)

            completed = manager._finish_temporal_observation(frame)

            self.assertTrue(completed)
            self.assertEqual("22K gold", manager._session.result)
            self.assertEqual("22K gold", manager._session.acid_result)
            self.assertTrue(manager._session.acid_ok)
            self.assertFalse(manager._baseline_crops)
            self.assertFalse(manager._transition_crops)
            self.assertFalse(manager._observation_crops)
            self.assertTrue(Path(manager._session.acid_success_image_path).is_file())
            self.assertTrue(Path(manager._session.acid_zoom_image_path).is_file())
            self.assertEqual(2, len(list(root.rglob("*.jpg"))))
            self.assertFalse(list(root.rglob("*.avi")))
            self.assertIn("Acid test completed. 22K gold.", spoken)

    def test_invalid_temporal_result_becomes_retry(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manager = self._manager(root)
            manager._module = SimpleNamespace(
                STATE={"stage": "OBSERVE_5S", "acid_positive_streak": 3}
            )
            manager._temporal_model = _FakeTemporalModel("invalid")
            manager._session.started_at = "started"
            manager._session.running = True
            frame = np.zeros((40, 50, 3), dtype=np.uint8)
            manager._baseline_crops = [frame.copy() for _ in range(5)]
            manager._transition_crops = [frame.copy() for _ in range(3)]
            manager._observation_crops = [frame.copy() for _ in range(10)]

            completed = manager._finish_temporal_observation(frame)

            self.assertFalse(completed)
            self.assertEqual("RETRY", manager.snapshot()["stage"])
            self.assertEqual("Inconclusive", manager.snapshot()["acid_result"])
            self.assertTrue(manager.snapshot()["acid_result_ready"])

    def test_detector_only_mode_preserves_existing_image_storage(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            manager = self._manager(root)
            manager.set_acid_result_enabled(False)
            manager._session_root = root / "purity"

            saved = manager._capture_session_image(
                "acid_success_image_path",
                "acid_ok",
                np.zeros((20, 20, 3), dtype=np.uint8),
            )

            self.assertTrue(saved)
            self.assertTrue(Path(saved).is_file())

    def test_reset_clears_temporal_state_and_previous_result(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manager = self._manager(Path(temporary_dir))
            frame = np.zeros((20, 20, 3), dtype=np.uint8)
            manager._baseline_crops = [frame]
            manager._transition_crops = [frame]
            manager._observation_crops = [frame]
            manager._reaction_bbox = (1, 2, 10, 12)
            manager._session.acid_result = "22K gold"

            manager.reset(stop_running=False)

            self.assertFalse(manager._baseline_crops)
            self.assertFalse(manager._transition_crops)
            self.assertFalse(manager._observation_crops)
            self.assertIsNone(manager._reaction_bbox)
            self.assertEqual("", manager.snapshot()["acid_result"])

    def test_full_temporal_state_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            frame = np.zeros((80, 100, 3), dtype=np.uint8)
            manager = PurityTestManager(
                base_dir=root,
                frame_getter=lambda: frame.copy(),
            )
            state = {
                "stage": "RUBBING",
                "rubbing_done": False,
                "rubbing_sync_hits": 0,
                "acid_positive_streak": 0,
                "stone_visible_now": True,
                "gold_visible_now": True,
                "sound_status": "OK",
                "audio_label": "OK",
                "audio_decision": "OK",
                "audio_confidence": 1.0,
                "last_rubbing_bbox": (5, 5, 20, 20),
                "last_rubbing_mask": None,
                "last_acid_bbox": None,
            }

            def process_acid(input_frame):
                bbox = (30, 25, 50, 40)
                state["last_acid_bbox"] = bbox
                return input_frame.copy(), True, {"acid_bbox": bbox, "error": ""}

            manager._module = SimpleNamespace(
                STATE=state,
                INFER_SKIP=1,
                RUBBING_SYNC_CONFIRM_FRAMES=1,
                ACID_CONFIRM_FRAMES=3,
                process_rubbing_frame=lambda input_frame: (
                    input_frame.copy(),
                    {"error": ""},
                ),
                compute_rubbing=lambda annotated, _info: (annotated, True),
                update_visual_rubbing_grace=lambda *_args, **_kwargs: None,
                process_acid_frame=process_acid,
                draw_status=lambda input_frame: input_frame,
            )
            manager._temporal_model = _FakeTemporalModel("non_gold")
            manager._session_root = root / "purity"
            manager._session.running = True
            manager._session.started_at = "started"
            manager._session.stage = "RUBBING"

            with (
                mock.patch("purity_test_manager.ACID_BASELINE_SECONDS", 0.05),
                mock.patch("purity_test_manager.ACID_OBSERVATION_SECONDS", 0.05),
                mock.patch("purity_test_manager.ACID_CAPTURE_FPS", 200.0),
                mock.patch("purity_test_manager.ACID_TRANSITION_CAPTURE_FPS", 200.0),
                mock.patch("purity_test_manager.ACID_BASELINE_MIN_FRAMES", 1),
                mock.patch("purity_test_manager.ACID_TRANSITION_MIN_FRAMES", 1),
                mock.patch("purity_test_manager.ACID_OBSERVATION_MIN_FRAMES", 1),
            ):
                worker = threading.Thread(target=manager._run_loop)
                worker.start()
                deadline = time.monotonic() + 2.0
                while manager.snapshot()["stage"] != "READY_FOR_ACID":
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                manager.start_acid_observation()
                worker.join(timeout=3.0)

            self.assertFalse(worker.is_alive())
            result = manager.snapshot()
            self.assertEqual("COMPLETED", result["stage"])
            self.assertEqual("Non-gold", result["acid_result"])
            self.assertFalse(result["running"])
            self.assertTrue(Path(result["rubbing_image_path"]).is_file())
            self.assertTrue(Path(result["rubbing_zoom_image_path"]).is_file())
            self.assertTrue(Path(result["acid_success_image_path"]).is_file())
            self.assertTrue(Path(result["acid_zoom_image_path"]).is_file())
            self.assertEqual(4, len(list(root.rglob("*.jpg"))))
            self.assertFalse(list(root.rglob("*.avi")))


if __name__ == "__main__":
    unittest.main()
