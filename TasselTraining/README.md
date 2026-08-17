# Tassel MobileNetV3 capture and training tool

This standalone PyQt6 application collects real tassel and hard-negative examples without changing the integrated jewellery workflow or its production checkpoint.

## Run

Only one process may own `/dev/video0`. Stop `integrated_ui_app.py` before starting this tool:

```bash
cd /home/pi/Desktop/EMBSYS-AI/TasselTraining
python app.py
```

The app loads `checkpoints/latest.pt` when it exists; otherwise it loads the current production checkpoint at `../Segmentation/tassel_mobilenet_v3_small.pt`.

## Capture and review

1. Choose **Draw Test Bed ROI**, then drag a green rectangle around only the test bed. The ROI is saved across restarts; **Clear Test Bed ROI** restores the full frame.
2. Position one Necklace or Haram in the ROI and choose **1. Capture Test Bed** after focus relocks.
3. Choose **2. Segment Tassel / Run Prediction**. The proposed tassel region is blue.
4. If the blue region is the real tassel, choose **3A. Blue Region is Correct Tassel**.
5. If the blue region is necklace, chain, pendant, thread that is not the tassel, or another wrong region, choose **3B. Blue Region is False Positive**.
6. If the model missed or incorrectly segmented a real tassel, paint the correct region in red with the left mouse button, erase with the right mouse button, then choose **3C. Save Red Drawn Tassel**. You can save 3B and 3C from the same capture.
7. If the image genuinely has no tassel, choose **3D. No Tassel in This Image**.
8. Choose **Return to Live Feed** and repeat.

The selected rectangle is also used for autofocus measurement. At capture time the camera image is cropped to the test bed before foreground extraction, candidate generation, MobileNet prediction, and training-label creation. Disconnected tassel/thread material is retained instead of keeping only the largest necklace component. Pixels outside the ROI cannot enter a saved training crop.

A false predicted candidate and a manually drawn real tassel may both be saved from the same capture. This provides the hard negative and the corrected positive without losing either example.

## Checkpoint training

Training requires at least eight `tassel` and eight `false_positive` samples. The UI shows exact shortfalls before launching the trainer.

**Train MobileNetV3 from Latest Checkpoint** always warm-starts from `checkpoints/latest.pt`, or from the production checkpoint on the first run. It never initializes random model weights and never overwrites the production model. It fine-tunes the last MobileNet feature blocks and classifier, keeps a timestamped checkpoint, updates `checkpoints/latest.pt`, writes a JSON training report, and reloads the new checkpoint in the app.

To evaluate the trained checkpoint in the integrated application, close this standalone tool, start `integrated_ui_app.py`, open **Setup Tools**, select **Latest Trained Model** under **Tassel Detection Model**, and choose **Apply Tassel Model**. Select **Production Model** to switch back. The selection is validated and persisted without overwriting either checkpoint.

Saved data:

```text
dataset/
  sessions/<capture_id>/
    camera_capture.png
    test_bed_crop.png
    working_image.png
    foreground_mask.png
    predicted_mask.png
    manual_tassel_mask.png
    manifest.json
  labels/
    tassel/*.png
    false_positive/*.png
checkpoints/
  latest.pt
  tassel_mobilenet_v3_<timestamp>.pt
  training_report_<timestamp>.json
```

## Tests

```bash
cd /home/pi/Desktop/EMBSYS-AI/TasselTraining
QT_QPA_PLATFORM=offscreen python -m unittest -v test_app.py
```
