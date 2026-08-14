import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app import (
    TRUE_DETECTION_THRESHOLD,
    DatasetStore,
    DetectionWorker,
    ModelProjectManager,
    classifier_training_shortfall,
    decode_detections,
    decode_yolov8_seg_rows,
    focus_scores_are_stable,
    is_single_class_yolov8_seg_shape,
    make_square_context_crop,
    normalize_nms_rows,
    normalize_legacy_manifest,
    project_id_from_model,
    roi_sharpness_score,
    set_camera_continuous_autofocus,
)
from ResNet18.train_resnet18 import (
    keep_frozen_batch_norm_eval,
    make_model,
    set_trainable_parameters,
    split_samples_by_crop,
)


class CropTests(unittest.TestCase):
    def test_default_crop_uses_five_percent_padding_and_square_letterbox(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        crop = make_square_context_crop(image, [20, 20, 40, 30])
        self.assertEqual(crop.shape, (22, 22, 3))
        self.assertTrue(np.any(crop == 114))
        self.assertTrue(np.any(crop == 255))

    def test_context_crop_is_square_and_pads_at_image_edge(self):
        image = np.full((80, 120, 3), 255, dtype=np.uint8)
        crop = make_square_context_crop(image, [0, 0, 20, 10], padding_ratio=0.20)
        self.assertEqual(crop.shape, (28, 28, 3))
        self.assertTrue(np.any(crop == 114))
        self.assertTrue(np.any(crop == 255))

    def test_nms_rows_and_coordinate_mapping(self):
        output = np.array([[[0.25, 0.25, 0.75, 0.75, 0.9]]], dtype=np.float32)
        rows = normalize_nms_rows(output)
        detections = decode_detections(rows, (640, 640), 1.0, 0, 0)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [160, 160, 480, 480])

    def test_multiclass_nms_rows_are_flattened_without_mixing_coordinates(self):
        output = np.zeros((2, 5, 2), dtype=np.float32)
        output[0, :, 0] = [0.10, 0.20, 0.30, 0.40, 0.90]
        output[1, :, 0] = [0.50, 0.60, 0.70, 0.80, 0.85]
        rows = normalize_nms_rows(output)
        detections = decode_detections(rows, (640, 640), 1.0, 0, 0)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0]["bbox"], [128, 64, 256, 192])
        self.assertEqual(detections[1]["bbox"], [384, 320, 512, 448])

    def test_adjustable_confidence_filters_candidates_below_point_five(self):
        rows = np.array(
            [
                [0.10, 0.10, 0.20, 0.20, 0.49],
                [0.30, 0.30, 0.40, 0.40, 0.50],
            ],
            dtype=np.float32,
        )
        detections = decode_detections(rows, (640, 640), 1.0, 0, 0, threshold=0.50)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0]["yolo_score"], 0.50)

    def test_raw_single_class_yolov8_seg_outputs_decode_to_bbox_rows(self):
        outputs = {"prototype": np.zeros((80, 80, 32), dtype=np.float32)}
        for size in (10, 20, 40):
            dfl = np.full((size, size, 64), -20.0, dtype=np.float32)
            scores = np.zeros((size, size, 1), dtype=np.float32)
            coefficients = np.zeros((size, size, 32), dtype=np.float32)
            for side in range(4):
                dfl[:, :, side * 16 + 1] = 20.0
            outputs[f"dfl_{size}"] = dfl
            outputs[f"scores_{size}"] = scores
            outputs[f"coefficients_{size}"] = coefficients
        outputs["scores_10"][4, 4, 0] = 0.90

        rows = decode_yolov8_seg_rows(outputs, (320, 320), threshold=0.50)
        detections = decode_detections(rows, (320, 320), 1.0, 0, 0, model_shape=(320, 320))

        self.assertEqual(rows.shape, (1, 5))
        self.assertEqual(detections[0]["bbox"], [112, 112, 176, 176])
        self.assertAlmostEqual(detections[0]["yolo_score"], 0.90)

    def test_bestnewacid_raw_output_shape_is_supported(self):
        shapes = []
        for size in (10, 20, 40):
            shapes.extend([(size, size, 64), (size, size, 1), (size, size, 32)])
        shapes.append((80, 80, 32))
        self.assertTrue(is_single_class_yolov8_seg_shape(shapes))

    def test_roi_sharpness_and_stability_gate(self):
        checker = np.indices((100, 100)).sum(axis=0) % 2
        sharp = np.repeat((checker * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
        roi = (10, 10, 90, 90)
        self.assertGreater(
            roi_sharpness_score(sharp, roi),
            roi_sharpness_score(blurred, roi),
        )
        self.assertTrue(focus_scores_are_stable([200.0] * 12))
        self.assertFalse(
            focus_scores_are_stable(
                [100.0, 120.0, 140.0, 160.0, 180.0, 200.0] * 2
            )
        )

    def test_continuous_autofocus_can_be_enabled_then_locked(self):
        class FakeCapture:
            autofocus = 0.0

            def set(self, property_id, value):
                self.assert_property = property_id
                self.autofocus = float(value)
                return True

            def get(self, property_id):
                self.assert_property = property_id
                return self.autofocus

        capture = FakeCapture()
        self.assertTrue(set_camera_continuous_autofocus(capture, True))
        self.assertEqual(capture.autofocus, 1.0)
        self.assertTrue(set_camera_continuous_autofocus(capture, False))
        self.assertEqual(capture.autofocus, 0.0)
        self.assertEqual(capture.assert_property, cv2.CAP_PROP_AUTOFOCUS)


class DatasetStoreTests(unittest.TestCase):
    def test_labels_persist_and_training_class_copy_tracks_latest_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DatasetStore(root)
            image = np.full((100, 100, 3), 180, dtype=np.uint8)
            manifest = store.create_session(
                image,
                [{"bbox": [20, 20, 60, 60], "yolo_score": 0.75}],
                "test.png",
            )
            session_id = manifest["session_id"]
            candidate_id = manifest["candidates"][0]["id"]
            labeled_name = f"{session_id}_{candidate_id}.png"

            store.set_label(session_id, candidate_id, "false_positive")
            self.assertTrue((root / "labels" / "false_positive" / labeled_name).exists())
            self.assertFalse((root / "labels" / "true_detection" / labeled_name).exists())

            store.set_label(session_id, candidate_id, "true_detection")
            self.assertFalse((root / "labels" / "false_positive" / labeled_name).exists())
            self.assertTrue((root / "labels" / "true_detection" / labeled_name).exists())
            saved = json.loads((root / "sessions" / session_id / "manifest.json").read_text())
            self.assertEqual(saved["candidates"][0]["label"], "true_detection")

    def test_resnet_result_draws_only_true_detection_boxes_and_saves_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DatasetStore(root)
            image = np.full((120, 120, 3), 180, dtype=np.uint8)
            manifest = store.create_session(
                image,
                [
                    {
                        "bbox": [20, 70, 40, 90],
                        "yolo_score": 0.80,
                        "resnet_prediction": "true_detection",
                        "resnet_confidence": 0.90,
                        "true_detection_probability": 0.90,
                    },
                    {
                        "bbox": [70, 70, 90, 90],
                        "yolo_score": 0.75,
                        "resnet_prediction": "false_positive",
                        "resnet_confidence": 0.85,
                        "true_detection_probability": 0.15,
                    },
                ],
                "test.png",
                analysis_mode="yolo_resnet_pt",
                resnet_model="test.pt",
            )

            annotated = cv2.imread(str(root / manifest["annotated_image"]), cv2.IMREAD_COLOR)
            self.assertEqual(manifest["yolo_candidate_count"], 2)
            self.assertEqual(manifest["detection_count"], 1)
            self.assertEqual(manifest["candidates"][0]["detection_number"], 1)
            self.assertNotIn("detection_number", manifest["candidates"][1])
            self.assertTrue(np.array_equal(annotated[70, 20], [0, 200, 0]))
            self.assertTrue(np.array_equal(annotated[70, 70], [180, 180, 180]))


class ModelProjectTests(unittest.TestCase):
    def test_training_shortfall_accounts_for_protected_evaluation_crop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels"
            for class_name, count in (("false_positive", 6), ("true_detection", 8)):
                class_dir = labels / class_name
                class_dir.mkdir(parents=True)
                for index in range(count):
                    (class_dir / f"sample_{index}.png").touch()

            counts, missing = classifier_training_shortfall(
                labels,
                root / "locked_crop_evaluation_manifest.json",
            )

            self.assertEqual(counts, {"false_positive": 6, "true_detection": 8})
            self.assertEqual(missing, {"false_positive": 2, "true_detection": 0})

    def test_imported_hefs_create_isolated_project_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_hef = root / "model one.hef"
            second_hef = root / "model_two.hef"
            first_hef.write_bytes(b"first")
            second_hef.write_bytes(b"second")
            inspector = lambda _path: {  # noqa: E731
                "input_height": 640,
                "input_width": 640,
                "input_channels": 3,
                "output_format": "FormatOrder.HAILO_NMS_BY_CLASS",
            }
            manager = ModelProjectManager(root / "projects", inspector=inspector)

            first = manager.import_hef(first_hef)
            second = manager.import_hef(second_hef)

            self.assertEqual(first["project_id"], "model_one")
            self.assertEqual(second["project_id"], "model_two")
            self.assertNotEqual(first["dataset_path"], second["dataset_path"])
            self.assertEqual(first["resnet_model_path"].name, "beadcheck.pt")
            self.assertEqual(first["candidate_model_path"].name, "beadcheck_candidate.pt")
            self.assertEqual(
                first["mobilenet_model_path"].name,
                "model_one_mobilenet_v3_candidate.pt",
            )
            self.assertNotEqual(first["mobilenet_model_path"], second["mobilenet_model_path"])
            self.assertTrue((first["dataset_path"] / "labels" / "true_detection").is_dir())
            self.assertEqual(len(manager.list_projects()), 2)

    def test_legacy_manifest_names_are_converted_without_changing_class_index(self):
        manifest = {
            "bead_count": 1,
            "candidates": [
                {
                    "label": "true_bead",
                    "resnet_prediction": "true_bead",
                    "true_bead_probability": 0.8,
                    "bead_number": 1,
                }
            ],
        }
        converted = normalize_legacy_manifest(manifest)
        self.assertEqual(converted["detection_count"], 1)
        self.assertEqual(converted["candidates"][0]["label"], "true_detection")
        self.assertEqual(converted["candidates"][0]["resnet_prediction"], "true_detection")
        self.assertEqual(converted["candidates"][0]["true_detection_probability"], 0.8)
        self.assertEqual(project_id_from_model(Path("model one.hef")), "model_one")

    def test_training_split_is_stratified_by_crop_not_session(self):
        samples = []
        for class_index, class_name in enumerate(("false_positive", "true_detection")):
            for index in range(20):
                samples.append(
                    {
                        "path": Path(f"{class_name}_{index}.png"),
                        "session_id": "same_session",
                        "class_name": class_name,
                        "class_index": class_index,
                    }
                )
        splits = split_samples_by_crop(samples, seed=42)
        self.assertEqual(
            {name: len(items) for name, items in splits.items()},
            {"train": 28, "validation": 6, "test": 6},
        )
        for items in splits.values():
            self.assertEqual({item["class_name"] for item in items}, set(("false_positive", "true_detection")))
        self.assertEqual(len({item["path"] for items in splits.values() for item in items}), 40)

    def test_classifier_acceptance_threshold_is_point_seven_five(self):
        self.assertEqual(TRUE_DETECTION_THRESHOLD, 0.75)

    def test_mobilenet_trains_only_last_features_and_classifier(self):
        model = make_model(pretrained=False, architecture="mobilenet_v3_small")
        trainable = set_trainable_parameters(model, "mobilenet_v3_small")
        model.train()
        keep_frozen_batch_norm_eval(model)
        self.assertTrue(trainable)
        self.assertFalse(model.features[0][0].weight.requires_grad)
        self.assertFalse(model.features[0][1].training)
        self.assertTrue(model.classifier[-1].weight.requires_grad)


class DetectionWorkerTests(unittest.TestCase):
    def test_worker_runs_yolo_on_the_captured_frame_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DatasetStore(root / "dataset")

            class FakeDetector:
                received = None
                threshold = None

                def detect(self, image, threshold):
                    self.received = image.copy()
                    self.threshold = threshold
                    return [{"bbox": [20, 20, 60, 60], "yolo_score": 0.75}]

            detector = FakeDetector()
            captured = np.full((100, 100, 3), 180, dtype=np.uint8)
            worker = DetectionWorker(
                detector,
                store,
                captured,
                "camera_capture.png",
                (10, 15, 90, 95),
                0.50,
            )
            captured[:] = 0
            completed = []
            failures = []
            worker.completed.connect(completed.append)
            worker.failed.connect(failures.append)
            worker.run()

            self.assertEqual(failures, [])
            self.assertEqual(len(completed), 1)
            self.assertTrue(np.all(detector.received == 180))
            self.assertEqual(detector.received.shape, (80, 80, 3))
            self.assertEqual(detector.threshold, 0.50)
            self.assertEqual(completed[0]["original_filename"], "camera_capture.png")
            self.assertEqual(completed[0]["roi"], [10, 15, 90, 95])
            self.assertEqual(completed[0]["yolo_threshold"], 0.50)
            self.assertEqual(completed[0]["candidates"][0]["bbox"], [30, 35, 70, 75])
            saved = cv2.imread(
                str(store.root / completed[0]["original_image"]),
                cv2.IMREAD_COLOR,
            )
            self.assertTrue(np.all(saved == 180))

    def test_worker_runs_yolo_before_resnet_and_saves_pt_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = DatasetStore(root / "dataset")
            events = []

            class FakeDetector:
                def detect(self, image, threshold):
                    events.append("yolo")
                    return [{"bbox": [10, 10, 50, 50], "yolo_score": 0.80}]

            class FakeClassifier:
                model_path = Path("beadcheck.pt")

                def predict_crops(self, crops):
                    events.append("resnet")
                    self.crops = crops
                    return [
                        {
                            "resnet_prediction": "true_detection",
                            "resnet_confidence": 0.90,
                            "true_detection_probability": 0.90,
                        }
                    ]

            classifier = FakeClassifier()
            captured = np.full((100, 100, 3), 180, dtype=np.uint8)
            worker = DetectionWorker(
                FakeDetector(),
                store,
                captured,
                "camera_capture.png",
                (20, 20, 80, 80),
                0.50,
                classifier=classifier,
            )
            completed = []
            failures = []
            worker.completed.connect(completed.append)
            worker.failed.connect(failures.append)
            worker.run()

            self.assertEqual(failures, [])
            self.assertEqual(events, ["yolo", "resnet"])
            self.assertEqual(len(classifier.crops), 1)
            self.assertEqual(completed[0]["analysis_mode"], "yolo_resnet_pt")
            self.assertEqual(completed[0]["resnet_model"], classifier.model_path.name)
            self.assertEqual(completed[0]["detection_count"], 1)
            candidate = completed[0]["candidates"][0]
            self.assertEqual(candidate["bbox"], [30, 30, 70, 70])
            self.assertEqual(candidate["resnet_prediction"], "true_detection")
            self.assertAlmostEqual(candidate["resnet_confidence"], 0.90)


if __name__ == "__main__":
    unittest.main()
