"""Generate a visual grid for checking training augmentations."""

from pathlib import Path

from PIL import Image, ImageDraw

from datasets.tsinghua_dogs import (
    IMAGENET_MEAN_RGB,
    TRAINING_AUGMENTATION,
    TRAIN_SPLIT_PATH,
    load_dog_crop,
    load_split_paths,
    resize_with_padding,
)

ML_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ML_ROOT / "data" / "tsinghua-dogs" / "augmentation_preview.jpg"

SAMPLE_INDEX = 0
AUGMENTED_VARIANTS = 8
GRID_COLUMNS = 3
CAPTION_HEIGHT = 24


def create_labeled_tile(image: Image.Image, label: str) -> Image.Image:
    """Place a caption above one model-sized image."""

    tile = Image.new(
        "RGB",
        (image.width, image.height + CAPTION_HEIGHT),
        IMAGENET_MEAN_RGB,
    )
    tile.paste(image, (0, CAPTION_HEIGHT))

    drawing = ImageDraw.Draw(tile)
    drawing.text(
        (8, 5),
        label,
        fill="white",
    )

    return tile


def main() -> None:
    """Save the original dog crop beside several randomized versions."""

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    dog_crop = load_dog_crop(training_paths[SAMPLE_INDEX])

    preview_images = [
        create_labeled_tile(
            resize_with_padding(dog_crop),
            "Original",
        )
    ]

    for variant_number in range(1, AUGMENTED_VARIANTS + 1):
        augmented_crop = TRAINING_AUGMENTATION(dog_crop.copy())
        model_input = resize_with_padding(augmented_crop)

        preview_images.append(
            create_labeled_tile(
                model_input,
                f"Variant {variant_number}",
            )
        )

    grid_rows = (len(preview_images) + GRID_COLUMNS - 1) // GRID_COLUMNS

    tile_width, tile_height = preview_images[0].size

    preview_grid = Image.new(
        "RGB",
        (GRID_COLUMNS * tile_width, grid_rows * tile_height),
        IMAGENET_MEAN_RGB,
    )

    for image_index, preview_image in enumerate(preview_images):
        column = image_index % GRID_COLUMNS
        row = image_index // GRID_COLUMNS

        preview_grid.paste(
            preview_image,
            (column * tile_width, row * tile_height),
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    preview_grid.save(OUTPUT_PATH, quality=95)

    print(f"Saved augmentation preview: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
