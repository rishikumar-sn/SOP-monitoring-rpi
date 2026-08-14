# HEF false-positive dataset tool

This is a standalone PyQt6 labeling and classifier-comparison tool. It captures a live camera frame, runs the active compatible detection HEF on that exact frame, and saves true-detection and false-positive review labels in a model-specific project.

## Run

Stop the main jewellery application first so the standalone process can own the Hailo device, then run:

```bash
cd /home/pi/Desktop/EMBSYS-AI/BeadFalsePositive
python app.py
```

The PyQt window opens directly. Camera index `0` is used by default; use `python app.py --camera 1` for another camera.

The default `models/bead_finder.hef` is imported as the `bead_finder` project. Use **Load HEF** to import another HEF. The tool accepts HEFs with a three-channel NHWC image input and either a Hailo NMS detection output or the single-class raw YOLOv8-seg output used by `models/bestnewacid.hef`. Other raw outputs, classification models, and super-resolution models are rejected because they require different decoders.

The camera is opened in MJPEG mode at the requested resolution. Continuous autofocus runs during the first 60 focus frames and until sharpness inside the selected ROI stabilizes; the application then locks autofocus so motion does not make the lens hunt. **Capture** remains disabled until the UI shows **Focus: locked**. Returning to **Live Feed** or drawing a new ROI starts and locks autofocus again.

1. Select the active HEF project or use **Load HEF**.
2. Position the object in the live camera feed.
3. Select **Draw ROI**, then drag a rectangle around the area the HEF should inspect.
4. Adjust **Confidence** if needed. The default is `0.50`.
5. Select **Capture**. The displayed frame is frozen and only its selected ROI is sent to the active HEF.
6. Review every crop as **True Detection**, **False Positive**, or **Unreviewed**.
7. Select **Live Feed** to return to the camera. The ROI stays active until **Clear ROI** is selected.

## Train and check with ResNet18

- Select **Train ResNet18** to train from the active project's labeled crops. Training warm-starts from the working model and writes `ResNet18/beadcheck_candidate.pt`; it never overwrites `beadcheck.pt`.
- **Check with ResNet and PT** tests the working model. **Check Candidate PT** tests the newly trained candidate on a frozen frame.
- The first protected crop training run creates `locked_crop_evaluation_manifest.json` and preserved evaluation images. Those exact crops are excluded from future training.
- **Promote Candidate** is enabled only when the candidate improves locked F0.5 without increasing false-positive acceptance.
- The result page and annotated image show only candidates ResNet18 accepts as true detections, with their bounding boxes and the final detection count. Rejected candidates are hidden from the result.

## Compare with MobileNetV3-Small

- Select **Train MobileNetV3** to create an isolated `MobileNetV3/<hef_name>_mobilenet_v3_candidate.pt` experiment. The original bead project retains its existing `beadcheck_mobilenet_v3_candidate.pt` filename. Training uses the same crop-stratified splits and protected crop evaluation as ResNet18.
- Select **Check HEF BBoxes + MobileNet** to run the HEF first, crop only its bounding boxes, and filter those crops with the experimental checkpoint.
- MobileNet training never overwrites `ResNet18/beadcheck.pt` or the production `models/beadcheck.pt` used by `integrated_ui_app.py`.
- Compare the locked accuracy, recall, F1, and confusion matrix in the training reports. A high ordinary test score with a low locked score indicates dataset/session leakage rather than a production-ready classifier.

Run `python compare_classifiers.py` to print both models' ordinary/locked metrics and dataset-health warnings. Use this after every new collection and training cycle.

New captures use only 5% context around each HEF box. Rectangular crops are padded with a neutral value to make them square before resizing, reducing the amount of chain or adjacent jewellery included in the classifier input. Existing dataset images are not rewritten.

The HEF candidate threshold defaults to `0.50`. A candidate is displayed and counted only when the PT classifier gives `true_detection` a probability of at least `0.75`. Training gives false-positive crops double loss weight so chain detections are penalized more strongly.

## Saved data

The tool creates one isolated folder per HEF filename:

```text
model_projects/<hef_name>/
  <hef_name>.hef
  project.json
  dataset/
    sessions/<session_id>/
      original.png
      annotated.png
      manifest.json
      crops/candidate_*.png
    labels/
      true_detection/*.png
      false_positive/*.png
  ResNet18/
    beadcheck.pt
    beadcheck_candidate.pt
    beadcheck_candidate_split_manifest.json
    beadcheck_candidate_training_report.json
    locked_crop_evaluation_manifest.json
    locked_crop_evaluation/
  MobileNetV3/
    <hef_name>_mobilenet_v3_candidate.pt
    <hef_name>_mobilenet_v3_candidate_split_manifest.json
    <hef_name>_mobilenet_v3_candidate_training_report.json
```

Every session manifest retains the HEF filename, source filename, ROI, selected confidence threshold, bounding box, detector score, crop padding, review time, and selected label. Switching projects switches the session list, training data, and PT model together.

Training uses the labeled crop files directly and makes deterministic, class-stratified train, validation, test, and protected evaluation sets. Sessions remain as dataset history but do not control training or promotion.

The original `dataset/` and `ResNet18/bead_fp_resnet18_test.pt` remain untouched. They are copied once into `model_projects/bead_finder`, with accepted labels renamed to `true_detection` and the copied checkpoint named `beadcheck.pt`. The integrated jewellery application uses its separate `models/beadcheck.pt`; standalone training does not change it.

## Tests

The tests do not access Hailo:

```bash
cd /home/pi/Desktop/EMBSYS-AI/BeadFalsePositive
python -m unittest -v test_app.py
```
