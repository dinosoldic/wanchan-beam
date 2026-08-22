"""Verify the FP32 mobile LiteRT detector against its PyTorch checkpoint."""

from pathlib import Path
from typing import cast

from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from evaluation.compare_mobile_detector_candidates import (
    CONFIDENCE_THRESHOLD,
    DOG_CLASS_ID,
    EXPECTED_DOG_COUNT,
    DogDetection,
    calculate_iou,
    suppress_duplicate_detections,
)

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent

PYTORCH_MODEL_PATH = ML_ROOT / "artifacts" / "detector" / "yolo26n.pt"
LITERT_MODEL_PATH = PROJECT_ROOT / "models" / "mobile" / "v1" / "detector.tflite"
SAMPLE_IMAGE_PATH = ML_ROOT / "data" / "samples" / "test-dogs.png"

INPUT_SIZE = 544

# These tolerances allow harmless floating-point drift while still detecting
# a meaningful conversion or output-decoding problem.
MAX_CONFIDENCE_DIFFERENCE = 0.05
MAX_COORDINATE_DIFFERENCE = 2.0
MINIMUM_BOX_IOU = 0.99


def run_detector(model_path: Path) -> list[DogDetection]:
    """Run one detector and return its retained dog detections."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Detector not found: {model_path}")

    model = YOLO(str(model_path), task="detect")
    results = cast(
        list[Results],
        model.predict(
            source=str(SAMPLE_IMAGE_PATH),
            imgsz=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[DOG_CLASS_ID],
            device="cpu",
            stream=False,
            verbose=False,
        ),
    )

    if len(results) != 1:
        raise RuntimeError(f"Expected one result, received {len(results)}")

    boxes = results[0].boxes

    if boxes is None:
        return []

    coordinates = cast(Tensor, boxes.xyxy).cpu().tolist()
    confidences = cast(Tensor, boxes.conf).cpu().tolist()

    raw_detections: list[DogDetection] = []

    for box, confidence in zip(coordinates, confidences, strict=True):
        x1, y1, x2, y2 = box

        raw_detections.append(
            DogDetection(
                confidence=float(confidence),
                box=(
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ),
            )
        )

    retained_detections = suppress_duplicate_detections(raw_detections)

    # The sample dogs are arranged horizontally, so center-X gives both
    # backends a stable spatial order independent of confidence ranking.
    return sorted(
        retained_detections,
        key=lambda detection: (detection.box[0] + detection.box[2]) / 2.0,
    )


def main() -> None:
    """Compare PyTorch and LiteRT detections on the reference image."""

    if not SAMPLE_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

    pytorch_detections = run_detector(PYTORCH_MODEL_PATH)
    litert_detections = run_detector(LITERT_MODEL_PATH)

    if len(pytorch_detections) != EXPECTED_DOG_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_DOG_COUNT} PyTorch detections, "
            f"received {len(pytorch_detections)}"
        )

    if len(litert_detections) != len(pytorch_detections):
        raise RuntimeError(
            f"Detection count differs: PyTorch={len(pytorch_detections)}, "
            f"LiteRT={len(litert_detections)}"
        )

    confidence_differences: list[float] = []
    coordinate_differences: list[float] = []
    box_ious: list[float] = []

    for index, (pytorch_detection, litert_detection) in enumerate(
        zip(pytorch_detections, litert_detections, strict=True),
        start=1,
    ):
        confidence_difference = abs(
            pytorch_detection.confidence - litert_detection.confidence
        )
        coordinate_difference = max(
            abs(pytorch_value - litert_value)
            for pytorch_value, litert_value in zip(
                pytorch_detection.box,
                litert_detection.box,
                strict=True,
            )
        )
        box_iou = calculate_iou(
            pytorch_detection.box,
            litert_detection.box,
        )

        confidence_differences.append(confidence_difference)
        coordinate_differences.append(coordinate_difference)
        box_ious.append(box_iou)

        print(
            f"Dog {index}: "
            f"PyTorch confidence={pytorch_detection.confidence:.6f}, "
            f"LiteRT confidence={litert_detection.confidence:.6f}, "
            f"maximum coordinate difference={coordinate_difference:.4f}px, "
            f"IoU={box_iou:.6f}"
        )

    maximum_confidence_difference = max(confidence_differences)
    maximum_coordinate_difference = max(coordinate_differences)
    minimum_box_iou = min(box_ious)

    print()
    print(f"Detections compared: {len(pytorch_detections)}")
    print("Maximum confidence difference: " f"{maximum_confidence_difference:.8f}")
    print("Maximum coordinate difference: " f"{maximum_coordinate_difference:.6f}px")
    print(f"Minimum box IoU: {minimum_box_iou:.8f}")

    if maximum_confidence_difference > MAX_CONFIDENCE_DIFFERENCE:
        raise RuntimeError("Confidence parity tolerance exceeded.")

    if maximum_coordinate_difference > MAX_COORDINATE_DIFFERENCE:
        raise RuntimeError("Coordinate parity tolerance exceeded.")

    if minimum_box_iou < MINIMUM_BOX_IOU:
        raise RuntimeError("Bounding-box IoU parity tolerance exceeded.")

    print("PyTorch/LiteRT mobile detector parity passed.")


if __name__ == "__main__":
    main()
