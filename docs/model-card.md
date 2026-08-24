# Model card

This model card records what the WanChan Beam models are expected to do, how
they were checked, and where their results should not be overinterpreted.

## Intended use

WanChan Beam detects visible dogs and estimates one of 130 breed labels for
each detected dog. It is intended for casual visual exploration in live camera
scenes and saved photos.

It is not intended to prove ancestry, identify an individual dog, diagnose a
medical condition, or replace genetic testing. Mixed breeds are outside the
classifier's single-label training target.

## Model package

| Role | Model | Runtime | Input | Output |
| --- | --- | --- | --- | --- |
| Server detector | YOLO26s | ONNX Runtime | `1 x 3 x 960 x 960` FP32 | `1 x 300 x 6` |
| Mobile detector | YOLO26n | LiteRT | `1 x 3 x 544 x 544` FP32 | `1 x 300 x 6` |
| Server classifier | MobileNetV3-Large | ONNX Runtime | `batch x 3 x 256 x 256` FP32 | `batch x 130` logits |
| Mobile classifier | MobileNetV3-Large | LiteRT | `1 x 3 x 256 x 256` FP32 | `1 x 130` logits |

The detector output fields are `x1`, `y1`, `x2`, `y2`, confidence, and class
ID. Only COCO class 16 (`dog`) is kept. Both detector exports are end-to-end,
but a conservative duplicate-box pass handles almost identical outputs that
can still describe the same dog.

The breed classifier uses a 1280-feature head with dropout `0.4`. Class ID is
the index of the breed in `models/v1/labels.json`; changing that order without
re-exporting the model would silently attach the wrong names.

## Preprocessing contract

Detector images are converted to RGB, letterboxed without changing aspect
ratio, padded with RGB `(114, 114, 114)`, normalized to `[0, 1]`, and stored in
NCHW order.

Breed crops:

1. Start from the detector box in original image coordinates.
2. Round the minimum edges down and maximum edges up.
3. Clamp the crop to the image.
4. Letterbox to `256 x 256` with RGB `(124, 116, 104)`.
5. Normalize with ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation
   `(0.229, 0.224, 0.225)`.

The server and mobile implementations deliberately reproduce the Python
rounding and channel order. Small differences here can lower accuracy even when
the exported model itself is correct.

## Breed training and evaluation

The classifier was trained on the Tsinghua Dogs dataset with 130 classes. The
reported validation split contains 5,200 images.

| Model stage | Top-1 | Top-2 |
| --- | ---: | ---: |
| Trained classifier head | 82.73% | 93.12% |
| Fine-tuned checkpoint | **85.04%** | **93.83%** |
| Improvement | **+2.31 points** | **+0.71 points** |

Top-1 requires the first prediction to match. Top-2 also accepts a correct
second prediction. The selected checkpoint was saved at epoch 4 with validation
loss `0.4872`.

These results use clean annotated crops. App accuracy can be lower because its
crops come from detector boxes and may include occlusion, background, unusual
poses, screens, or only part of a dog.

## Confidence behavior

The clean validation split produced this threshold tradeoff:

| Threshold | Coverage | Accuracy of shown labels |
| ---: | ---: | ---: |
| 0.30 | 98.13% | 86.11% |
| 0.35 | 96.46% | 86.94% |
| 0.40 | 94.58% | 87.88% |
| 0.45 | 92.23% | 88.87% |
| 0.50 | 88.88% | 90.46% |

The app currently uses `0.40` for live labels and `0.34` for photo labels.
Those are product thresholds, not proof that a displayed percentage is a
perfect probability. The photo threshold is lower because detector-generated
crops are less controlled than validation crops.

A stronger future calibration set should contain manually labelled crops from
real app photos, including multiple dogs, partial bodies, rotation, and poor
lighting.

## Export parity

Export checks compare the deployed runtimes against their PyTorch sources.

### Breed classifier

- ONNX: all 5,200 validation images kept the same top-1 and ordered top-2 IDs.
- ONNX maximum absolute logit difference: `0.0000394583`.
- LiteRT: all 5,200 validation images kept the same top-1 and ordered top-2 IDs.
- LiteRT maximum absolute logit difference: `0.00005126`.

### Mobile detector

On the six-dog reference image, PyTorch and LiteRT returned six matching
detections. The minimum box IoU was `0.9936216` and the maximum coordinate
difference was `1.05777` pixels.

This detector parity result covers one difficult reference scene. It verifies
the export path, not general detector accuracy.

## Known limitations

- Similar breeds can have low separation, especially related terriers, poodles,
  shepherds, collies, and spitz-type breeds.
- Lying, upside-down, distant, or partly hidden dogs can produce low confidence.
- Detector errors become classifier errors because the classifier only sees the
  detector crop.
- The classifier must choose from its 130 trained labels; it has no explicit
  mixed-breed or out-of-distribution class.
- Live tracking is positional and may swap track IDs when dogs cross.
- Validation accuracy is not the same as accuracy on real camera scenes.

The UI uses `Breed uncertain` below its display threshold instead of presenting
every top prediction as reliable.

## Reproducibility files

- `models/v1/manifest.json` records mobile/server artifact hashes and export
  metadata.
- `models/v1/preprocessing.json` is the preprocessing contract.
- `ml/evaluation/evaluate_breed_classifier.py` recalculates validation metrics
  and confidence coverage.
- `ml/export/verify_breed_classifier_onnx.py` checks server classifier parity.
- `ml/export/verify_mobile_breed_classifier_litert.py` checks mobile classifier
  parity.
- `ml/export/verify_mobile_detector_litert.py` checks mobile detector parity.

Dataset and upstream model citations are listed in the main
[README](../README.md#data-models-and-attribution).
