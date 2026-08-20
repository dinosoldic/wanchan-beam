"""Reusable helpers and dataset loader for Tsinghua Dogs."""

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

ML_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ML_ROOT / "data" / "tsinghua-dogs" / "raw"
SPLIT_ROOT = DATASET_ROOT / "TrainAndValList"
IMAGE_ROOT = DATASET_ROOT / "low-resolution"
ANNOTATION_ROOT = DATASET_ROOT / "Low-Annotations"

TRAIN_SPLIT_PATH = SPLIT_ROOT / "train.lst"
VALIDATION_SPLIT_PATH = SPLIT_ROOT / "validation.lst"

MODEL_INPUT_SIZE = 224
IMAGENET_MEAN_RGB = (124, 116, 104)
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION: list[float] = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class BoundingBox:
    """Store the pixel coordinates of one rectangular dog annotation."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin


class TsinghuaDogsDataset(Dataset[tuple[Tensor, int]]):
    """Load Tsinghua dog crops and their numeric breed labels on demand."""

    def __init__(
        self,
        image_paths: list[Path],
        breed_to_id: dict[str, int],
        image_augmentation: Callable[[Image.Image], Image.Image] | None = None,
    ) -> None:
        self.image_paths = tuple(image_paths)
        self.breed_to_id = breed_to_id
        self.image_augmentation = image_augmentation

    def __len__(self) -> int:
        """Return the number of available training examples."""

        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and preprocess one dog using its dataset index."""

        relative_image_path = self.image_paths[index]
        breed_name = get_breed_name(relative_image_path)
        breed_id = self.breed_to_id[breed_name]

        dog_crop = load_dog_crop(relative_image_path)

        if self.image_augmentation is not None:
            dog_crop = self.image_augmentation(dog_crop)

        model_input = resize_with_padding(dog_crop)
        image_tensor = image_to_normalized_tensor(model_input)

        return image_tensor, breed_id


def build_training_augmentation(
    *,
    use_geometric_augmentation: bool,
) -> transforms.Compose:
    """Build the training transforms with optional geometric variation."""

    augmentation_steps: list[nn.Module] = [
        transforms.RandomHorizontalFlip(p=0.5),
    ]

    if use_geometric_augmentation:
        augmentation_steps.append(
            transforms.RandomApply(
                [
                    transforms.RandomAffine(
                        degrees=8,
                        translate=(0.04, 0.04),
                        scale=(0.95, 1.05),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=IMAGENET_MEAN_RGB,  # pyright: ignore[reportArgumentType]
                    ),
                ],
                p=0.5,
            )
        )

    augmentation_steps.append(
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )
    )

    return transforms.Compose(augmentation_steps)


def load_split_paths(split_path: Path) -> list[Path]:
    """Load dataset-relative image paths from an official split file."""
    image_paths: list[Path] = []

    # utf-8-sig removes the hidden BOM present at the start of these files.
    for raw_line in split_path.read_text(encoding="utf-8-sig").splitlines():
        relative_path = raw_line.strip().removeprefix(".//")

        if relative_path:
            image_paths.append(Path(relative_path))

    return image_paths


def require_coordinate(
    box_element: ET.Element,
    coordinate_name: str,
    annotation_path: Path,
) -> int:
    """Read a required bounding-box coordinate from an annotation."""

    value = box_element.findtext(coordinate_name)

    if value is None:
        raise ValueError(
            f"Missing {coordinate_name!r} in annotation: {annotation_path}"
        )

    return int(value)


def load_body_box(relative_image_path: Path) -> BoundingBox:
    """Load the whole-body box for one dog from its XML annotation."""

    annotation_path = ANNOTATION_ROOT / f"{relative_image_path}.xml"
    root = ET.parse(annotation_path).getroot()
    body_box_element = root.find("object/bodybndbox")

    if body_box_element is None:
        raise ValueError(f"Missing body bounding box in annotation: {annotation_path}")

    return BoundingBox(
        xmin=require_coordinate(body_box_element, "xmin", annotation_path),
        ymin=require_coordinate(body_box_element, "ymin", annotation_path),
        xmax=require_coordinate(body_box_element, "xmax", annotation_path),
        ymax=require_coordinate(body_box_element, "ymax", annotation_path),
    )


def clamp_box(
    box: BoundingBox,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Restrict a box to the real decoded image boundaries."""

    return BoundingBox(
        xmin=min(max(box.xmin, 0), image_width),
        ymin=min(max(box.ymin, 0), image_height),
        xmax=min(max(box.xmax, 0), image_width),
        ymax=min(max(box.ymax, 0), image_height),
    )


def load_dog_crop(relative_image_path: Path) -> Image.Image:
    """Open an image and return its annotated whole-dog crop in RGB."""

    image_path = IMAGE_ROOT / relative_image_path
    body_box = load_body_box(relative_image_path)

    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

        # JPEG dimensions are authoritative because some XML sizes are swapped.
        clamped_box = clamp_box(body_box, *rgb_image.size)

        # Fall back to the complete image when the annotation has no usable area.
        if clamped_box.width <= 0 or clamped_box.height <= 0:
            return rgb_image.copy()

        return rgb_image.crop(
            (
                clamped_box.xmin,
                clamped_box.ymin,
                clamped_box.xmax,
                clamped_box.ymax,
            )
        )


def get_breed_name(relative_image_path: Path) -> str:
    """Extract the breed name from its official dataset directory name."""
    breed_directory = relative_image_path.parent.name
    directory_parts = breed_directory.split("-", maxsplit=2)

    if len(directory_parts) != 3:
        raise ValueError(f"Unexpected breed directory name: {breed_directory}")

    return directory_parts[2]


def build_breed_mapping(
    image_paths: list[Path],
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Create stable conversions between breed names and model output IDs."""
    breed_names = tuple(
        sorted({get_breed_name(image_path) for image_path in image_paths})
    )

    if len(breed_names) != 130:
        raise ValueError(f"Expected 130 breeds, but found {len(breed_names)}")

    # Sorting above guarantees that every run assigns the same numeric IDs.
    breed_to_id = {
        breed_name: breed_id for breed_id, breed_name in enumerate(breed_names)
    }

    return breed_names, breed_to_id


def resize_with_padding(
    image: Image.Image,
    output_size: int = MODEL_INPUT_SIZE,
) -> Image.Image:
    """Resize an image proportionally and center it on a square canvas."""

    original_width, original_height = image.size
    scale = min(
        output_size / original_width,
        output_size / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized_image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    square_image = Image.new(
        "RGB",
        (output_size, output_size),
        IMAGENET_MEAN_RGB,
    )

    padding_left = (output_size - resized_width) // 2
    padding_top = (output_size - resized_height) // 2

    square_image.paste(
        resized_image,
        (padding_left, padding_top),
    )

    return square_image


def image_to_normalized_tensor(image: Image.Image) -> Tensor:
    """Convert an RGB image into the normalized tensor MobileNet expects."""

    image_tensor = transform_functional.to_tensor(image)

    return transform_functional.normalize(
        image_tensor,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STANDARD_DEVIATION,
    )


if __name__ == "__main__":
    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    train_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(train_paths)
    sample_path = train_paths[0]
    sample_crop = load_dog_crop(sample_path)

    preview_path = ML_ROOT / "data" / "tsinghua-dogs" / "sample_body_crop.jpg"
    sample_crop.save(preview_path, quality=95)

    model_input = resize_with_padding(sample_crop)
    model_input_path = ML_ROOT / "data" / "tsinghua-dogs" / "sample_model_input.jpg"
    model_input.save(model_input_path, quality=95)

    model_tensor = image_to_normalized_tensor(model_input)

    training_augmentation = build_training_augmentation(
        use_geometric_augmentation=False,
    )

    train_dataset = TsinghuaDogsDataset(
        train_paths,
        breed_to_id,
        image_augmentation=training_augmentation,
    )

    dataset_tensor, dataset_breed_id = train_dataset[0]

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch_images, batch_breed_ids = next(iter(train_data_loader))

    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    validation_images, validation_breed_ids = next(iter(validation_data_loader))

    # print(f"Training images: {len(train_paths):,}")
    # print(f"Breeds: {len(breed_names)}")
    # print(f"Shiba Dog ID: {breed_to_id['Shiba_Dog']}")
    # print(f"Breed at that ID: {breed_names[breed_to_id['Shiba_Dog']]}")

    # print(f"Sample crop size: {sample_crop.size}")
    # print(f"Saved sample crop: {preview_path}")

    # print(f"Model input size: {model_input.size}")
    # print(f"Saved model input: {model_input_path}")

    # print(f"Tensor shape: {model_tensor.shape}")
    # print(f"Tensor type: {model_tensor.dtype}")

    # print(f"Dataset samples: {len(train_dataset):,}")
    # print(f"Dataset tensor shape: {dataset_tensor.shape}")
    # print(f"Dataset breed ID: {dataset_breed_id}")
    # print(f"Dataset breed: {breed_names[dataset_breed_id]}")

    # print(f"Batch image shape: {batch_images.shape}")
    # print(f"Batch label shape: {batch_breed_ids.shape}")
    # print(f"Batch breed IDs: {batch_breed_ids}")

    # print(f"Validation samples: {len(validation_dataset):,}")
    # print(f"Validation image shape: {validation_images.shape}")
    # print(f"Validation label shape: {validation_breed_ids.shape}")
    # print(f"Validation breed IDs: {validation_breed_ids}")
