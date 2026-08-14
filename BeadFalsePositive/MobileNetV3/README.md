# MobileNetV3-Small classifier experiment

Use **Train MobileNetV3** in the standalone PyQt application. The experiment
uses PyTorch and Torchvision already available to `BeadFalsePositive`; it does
not require a separate package installation or change the OCR environment.

The trainer shares the ResNet crop-stratified split and locked crop evaluation, but
saves separate artifacts under the active HEF project's `MobileNetV3/` folder. For
example, `bestnewacid.hef` writes
`model_projects/bestnewacid/MobileNetV3/bestnewacid_mobilenet_v3_candidate.pt`.
It fine-tunes the last three MobileNet feature blocks and classifier from
ImageNet weights.

The production jewellery application is not changed automatically. Treat the
locked metrics, false-positive acceptance rate, and confusion matrix as
authoritative.
