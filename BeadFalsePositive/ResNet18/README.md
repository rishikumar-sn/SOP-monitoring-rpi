# Detection false-positive ResNet-18

This shared trainer trains a ResNet18 or MobileNetV3-Small binary classifier from the active project's
`dataset/labels/{false_positive,true_detection}` folders.

The current dataset is intentionally treated as a proof of the training and
export pipeline. Do not integrate its output into the production jewellery
workflow until substantially more balanced, varied data is collected.

## Train

Use **Train ResNet18** in the main PyQt application, or run the same trainer directly:

```bash
cd /home/pi/Desktop/EMBSYS-AI/BeadFalsePositive/ResNet18
python train_resnet18.py
```

The standalone UI passes `--architecture mobilenet_v3_small` for its isolated
MobileNet comparison; ResNet18 remains the command-line default.

Outputs:

- `../model_projects/bead_finder/ResNet18/beadcheck_candidate.pt`: candidate raw `state_dict`
- `../model_projects/bead_finder/ResNet18/beadcheck_candidate_split_manifest.json`: candidate assignments
- `../model_projects/bead_finder/ResNet18/beadcheck_candidate_training_report.json`: candidate and baseline metrics
- `../model_projects/bead_finder/ResNet18/locked_crop_evaluation_manifest.json`: fixed crop-based promotion reference

The class mapping matches the existing false-positive filter:

- `false_positive = 0`
- `true_detection = 1`

Training warm-starts from the working `beadcheck.pt` when it exists. It never
overwrites that working checkpoint. Training is stratified by individual
labeled crops, not capture sessions. The candidate must improve locked F0.5
without increasing false-positive acceptance before the UI enables
**Promote Candidate**.

The application's **Check with ResNet and PT** action runs the active HEF first
and then loads that project's checkpoint to classify each candidate crop. The
PyQt UI passes explicit dataset and output paths when another HEF project is active.
