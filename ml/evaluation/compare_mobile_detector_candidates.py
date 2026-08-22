"""Compare mobile detector input-size candidates on the six-dog sample."""

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

ML_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ML_ROOT / "artifacts" / "detector" / "yolo26n.pt"
SAMPLE_IMAGE_PATH = ML_ROOT / "data" / "samples" / "test-dogs.png"

INPUT_SIZES = (512, 544, 576, 608, 640)
DOG_CLASS_ID = 16
CONFIDENCE_THRESHOLD = 0.15
EXPECTED_DOG_COUNT = 6

DUPLICATE_IOU_THRESHOLD = 0.85
MAX_CENTER_OFFSET_RATIO = 0.15
DUPLICATE_CONTAINMENT_THRESHOLD = 0.98
MATCHING_EDGE_OFFSET_RATIO = 0.03
MINIMUM_MATCHING_EDGE_COUNT = 3

DetectionBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class DogDetection:
    """One dog candidate returned by the detector."""

    confidence: float
    box: DetectionBox


def get_box_width(box: DetectionBox) -> float:
    return box[2] - box[0]


def get_box_height(box: DetectionBox) -> float:
    return box[3] - box[1]


def get_box_area(box: DetectionBox) -> float:
    return get_box_width(box) * get_box_height(box)


def get_intersection_area(
    first_box: DetectionBox,
    second_box: DetectionBox,
) -> float:
    intersection_width = max(
        0.0,
        min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]),
    )
    intersection_height = max(
        0.0,
        min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]),
    )

    return intersection_width * intersection_height


def calculate_iou(first_box: DetectionBox, second_box: DetectionBox) -> float:
    intersection_area = get_intersection_area(first_box, second_box)
    union_area = get_box_area(first_box) + get_box_area(second_box) - intersection_area

    return intersection_area / union_area if union_area > 0.0 else 0.0


def calculate_containment(
    first_box: DetectionBox,
    second_box: DetectionBox,
) -> float:
    smaller_area = min(get_box_area(first_box), get_box_area(second_box))

    if smaller_area <= 0.0:
        return 0.0

    return get_intersection_area(first_box, second_box) / smaller_area


def have_similar_centers(
    first_box: DetectionBox,
    second_box: DetectionBox,
) -> bool:
    first_center_x = (first_box[0] + first_box[2]) / 2.0
    first_center_y = (first_box[1] + first_box[3]) / 2.0
    second_center_x = (second_box[0] + second_box[2]) / 2.0
    second_center_y = (second_box[1] + second_box[3]) / 2.0

    reference_width = min(get_box_width(first_box), get_box_width(second_box))
    reference_height = min(
        get_box_height(first_box),
        get_box_height(second_box),
    )

    return (
        abs(first_center_x - second_center_x)
        <= reference_width * MAX_CENTER_OFFSET_RATIO
        and abs(first_center_y - second_center_y)
        <= reference_height * MAX_CENTER_OFFSET_RATIO
    )


def have_at_least_three_matching_edges(
    first_box: DetectionBox,
    second_box: DetectionBox,
) -> bool:
    horizontal_tolerance = (
        min(get_box_width(first_box), get_box_width(second_box))
        * MATCHING_EDGE_OFFSET_RATIO
    )
    vertical_tolerance = (
        min(get_box_height(first_box), get_box_height(second_box))
        * MATCHING_EDGE_OFFSET_RATIO
    )

    matching_edges = (
        abs(first_box[0] - second_box[0]) <= horizontal_tolerance,
        abs(first_box[2] - second_box[2]) <= horizontal_tolerance,
        abs(first_box[1] - second_box[1]) <= vertical_tolerance,
        abs(first_box[3] - second_box[3]) <= vertical_tolerance,
    )

    return sum(matching_edges) >= MINIMUM_MATCHING_EDGE_COUNT


def suppress_duplicate_detections(
    detections: list[DogDetection],
) -> list[DogDetection]:
    """Match the server's duplicate-removal policy for dog boxes."""

    detections_by_confidence = sorted(
        detections,
        key=lambda detection: detection.confidence,
        reverse=True,
    )
    kept_detections: list[DogDetection] = []

    for candidate in detections_by_confidence:
        is_duplicate = any(
            (
                calculate_iou(candidate.box, kept.box) >= DUPLICATE_IOU_THRESHOLD
                and have_similar_centers(candidate.box, kept.box)
            )
            or (
                calculate_containment(candidate.box, kept.box)
                >= DUPLICATE_CONTAINMENT_THRESHOLD
                and have_at_least_three_matching_edges(candidate.box, kept.box)
            )
            for kept in kept_detections
        )

        if not is_duplicate:
            kept_detections.append(candidate)

    return kept_detections


def main() -> None:
    """Run the nano detector at each candidate input size."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Detector not found: {MODEL_PATH}")

    if not SAMPLE_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMAGE_PATH}")

    model = YOLO(str(MODEL_PATH))

    print(f"Model: {MODEL_PATH}")
    print(f"Sample: {SAMPLE_IMAGE_PATH}")
    print("CPU timings are for relative comparison only.")
    print()

    for input_size in INPUT_SIZES:
        results = cast(
            list[Results],
            model.predict(
                source=str(SAMPLE_IMAGE_PATH),
                imgsz=input_size,
                conf=CONFIDENCE_THRESHOLD,
                classes=[DOG_CLASS_ID],
                device="cpu",
                stream=False,
                verbose=False,
            ),
        )

        if len(results) != 1:
            raise RuntimeError(
                f"Expected one result at input size {input_size}, "
                f"received {len(results)}",
            )

        result = results[0]
        boxes = result.boxes

        if boxes is None:
            raw_detections: list[DogDetection] = []
        else:
            coordinate_tensor = cast(Tensor, boxes.xyxy)
            confidence_tensor = cast(Tensor, boxes.conf)
            coordinates = coordinate_tensor.cpu().tolist()
            confidences = confidence_tensor.cpu().tolist()
            raw_detections = []

            for box_coordinates, confidence in zip(
                coordinates,
                confidences,
                strict=True,
            ):
                x1, y1, x2, y2 = box_coordinates

                raw_detections.append(
                    DogDetection(
                        confidence=float(confidence),
                        box=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        kept_detections = suppress_duplicate_detections(raw_detections)

        raw_inference_ms = result.speed.get("inference")
        inference_ms = (
            float("nan") if raw_inference_ms is None else float(raw_inference_ms)
        )
        status = "PASS" if len(kept_detections) == EXPECTED_DOG_COUNT else "REVIEW"

        formatted_confidences = ", ".join(
            f"{detection.confidence:.3f}" for detection in kept_detections
        )

        print(
            f"{input_size:>3}x{input_size}: "
            f"raw={len(raw_detections)}, "
            f"kept={len(kept_detections)}, "
            f"inference={inference_ms:.1f}ms, "
            f"status={status}"
        )
        print(f"  confidences: [{formatted_confidences}]")

        for detection in kept_detections:
            x1, y1, x2, y2 = detection.box

            print(
                f"  {detection.confidence:.3f}: "
                f"({x1:.1f}, {y1:.1f})-({x2:.1f}, {y2:.1f})"
            )


if __name__ == "__main__":
    main()
