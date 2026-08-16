import argparse
import json
import math
from pathlib import Path
from typing import cast

import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

ML_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ML_ROOT / "artifacts" / "detector" / "yolo26s.pt"
OUTPUT_ROOT = ML_ROOT / "data" / "processed" / "dog-crops"

CROP_PADDING_RATIO = 0.05

INPUT_SIZE = 960
CONFIDENCE_THRESHOLD = 0.15
DOG_CLASS_ID = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect dogs in one image.")
    parser.add_argument(
        "image",
        type=Path,
        help="Path to the input image.",
    )
    return parser.parse_args()


def expand_box(
    coordinates: list[float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = coordinates

    padding_x = (x2 - x1) * CROP_PADDING_RATIO
    padding_y = (y2 - y1) * CROP_PADDING_RATIO

    return (
        max(0, math.floor(x1 - padding_x)),
        max(0, math.floor(y1 - padding_y)),
        min(image_width, math.ceil(x2 + padding_x)),
        min(image_height, math.ceil(y2 + padding_y)),
    )


def main() -> None:
    args = parse_args()
    image_path = args.image.resolve()

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Detector not found: {MODEL_PATH}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = YOLO(str(MODEL_PATH))

    results = cast(
        list[Results],
        model.predict(
            source=str(image_path),
            imgsz=INPUT_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=[DOG_CLASS_ID],
            stream=False,
            verbose=False,
        ),
    )

    result = results[0]
    boxes = result.boxes

    if boxes is None:
        print("Detected dogs: 0")
        return

    original_image = result.orig_img
    image_height, image_width = original_image.shape[:2]

    output_dir = OUTPUT_ROOT / image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detected dogs: {len(boxes)}")

    boxes_cpu = boxes.cpu()
    detections: list[dict[str, object]] = []

    for index, (coordinates, confidence) in enumerate(
        zip(boxes_cpu.xyxy.tolist(), boxes_cpu.conf.tolist()),
        start=1,
    ):
        x1, y1, x2, y2 = coordinates

        print(
            f"Dog {index}: "
            f"box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
            f"confidence={confidence:.3f}"
        )

        crop_x1, crop_y1, crop_x2, crop_y2 = expand_box(
            coordinates,
            image_width,
            image_height,
        )

        crop = original_image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_path = output_dir / f"dog-{index:02d}.png"

        if not cv2.imwrite(str(crop_path), crop):
            raise OSError(f"Could not save crop: {crop_path}")

        print(f"Saved crop: {crop_path}")

        detections.append(
            {
                "index": index,
                "class_id": DOG_CLASS_ID,
                "class_name": "dog",
                "confidence": round(float(confidence), 6),
                "bbox_xyxy": [round(float(value), 2) for value in coordinates],
                "crop_bbox_xyxy": [
                    crop_x1,
                    crop_y1,
                    crop_x2,
                    crop_y2,
                ],
                "crop_file": crop_path.name,
            }
        )

    manifest = {
        "source_image": str(image_path),
        "image_size": {
            "width": image_width,
            "height": image_height,
        },
        "detector": {
            "model": MODEL_PATH.name,
            "input_size": INPUT_SIZE,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "class_filter": [DOG_CLASS_ID],
            "crop_padding_ratio": CROP_PADDING_RATIO,
        },
        "detections": detections,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
