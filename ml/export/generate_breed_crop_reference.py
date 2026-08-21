"""Generate a Pillow reference crop for server preprocessing parity."""

from math import ceil, floor
from pathlib import Path

from PIL import Image

from datasets.tsinghua_dogs import resize_with_padding

ML_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ML_ROOT.parent

SOURCE_IMAGE_PATH = ML_ROOT / "data" / "samples" / "test-dogs.png"

OUTPUT_IMAGE_PATH = (
    REPOSITORY_ROOT
    / "server"
    / "tests"
    / "fixtures"
    / "breed-crop-python-reference.png"
)

INPUT_SIZE = 256

DETECTION_BOX = {
    "x1": 565.9,
    "y1": 131.5,
    "x2": 774.5,
    "y2": 422.5,
}


def clamp(value: int, minimum: int, maximum: int) -> int:
    """Restrict an integer coordinate to an image boundary."""

    return min(max(value, minimum), maximum)


def main() -> None:
    """Crop the selected dog and save its training-style model input."""

    if not SOURCE_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMAGE_PATH}")

    with Image.open(SOURCE_IMAGE_PATH) as image:
        rgb_image = image.convert("RGB")
        image_width, image_height = rgb_image.size

        # Match the server's outward rounding for detector coordinates.
        crop_left = clamp(
            floor(DETECTION_BOX["x1"]),
            0,
            image_width,
        )
        crop_top = clamp(
            floor(DETECTION_BOX["y1"]),
            0,
            image_height,
        )
        crop_right = clamp(
            ceil(DETECTION_BOX["x2"]),
            0,
            image_width,
        )
        crop_bottom = clamp(
            ceil(DETECTION_BOX["y2"]),
            0,
            image_height,
        )

        if crop_right <= crop_left or crop_bottom <= crop_top:
            raise ValueError("Reference crop has no usable area")

        dog_crop = rgb_image.crop(
            (
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
            )
        )

        model_input = resize_with_padding(
            dog_crop,
            output_size=INPUT_SIZE,
        )

        OUTPUT_IMAGE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        model_input.save(
            OUTPUT_IMAGE_PATH,
            format="PNG",
        )

    print(f"Source image: {SOURCE_IMAGE_PATH}")
    print(
        "Rounded crop box: "
        f"({crop_left}, {crop_top}) "
        f"({crop_right}, {crop_bottom})"
    )
    print(f"Crop size: " f"{crop_right - crop_left}x" f"{crop_bottom - crop_top}")
    print(f"Reference size: {model_input.size}")
    print(f"Saved reference: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
