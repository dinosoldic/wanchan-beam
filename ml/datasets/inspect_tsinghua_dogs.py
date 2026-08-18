"""Inspect the raw Tsinghua Dogs splits and annotation quality."""

from pathlib import Path
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from collections import Counter

ML_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ML_ROOT / "data" / "tsinghua-dogs" / "raw"
ANNOTATION_ROOT = DATASET_ROOT / "Low-Annotations"
SPLIT_ROOT = DATASET_ROOT / "TrainAndValList"

TRAIN_SPLIT_PATH = SPLIT_ROOT / "train.lst"
VALIDATION_SPLIT_PATH = SPLIT_ROOT / "validation.lst"


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        """Return the box width in pixels."""
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        """Return the box height in pixels."""
        return self.ymax - self.ymin


@dataclass(frozen=True)
class AnnotatedDog:
    breed: str
    body_box: BoundingBox
    head_box: BoundingBox


@dataclass(frozen=True)
class ImageAnnotation:
    width: int
    height: int
    dogs: tuple[AnnotatedDog, ...]


def load_split_paths(split_path: Path) -> list[Path]:
    """Load dataset-relative image paths from an official split file."""
    image_paths: list[Path] = []

    # utf-8-sig removes the hidden BOM present at the start of these files.
    for raw_line in split_path.read_text(encoding="utf-8-sig").splitlines():
        relative_path = raw_line.strip().removeprefix(".//")

        if relative_path:
            image_paths.append(Path(relative_path))

    return image_paths


def require_integer(
    element: ET.Element,
    field_name: str,
    annotation_path: Path,
) -> int:
    """Read a required integer field or report which annotation is malformed."""
    value = element.findtext(field_name)

    if value is None:
        raise ValueError(f"Missing {field_name!r} in annotation: {annotation_path}")

    return int(value)


def require_element(
    parent: ET.Element,
    element_name: str,
    annotation_path: Path,
) -> ET.Element:
    """Return a required XML child or report which annotation is malformed."""
    element = parent.find(element_name)

    if element is None:
        raise ValueError(f"Missing {element_name!r} in annotation: {annotation_path}")

    return element


def parse_box(
    box_element: ET.Element,
    annotation_path: Path,
) -> BoundingBox:
    """Convert one XML bounding-box element into a typed box."""
    return BoundingBox(
        xmin=require_integer(box_element, "xmin", annotation_path),
        ymin=require_integer(box_element, "ymin", annotation_path),
        xmax=require_integer(box_element, "xmax", annotation_path),
        ymax=require_integer(box_element, "ymax", annotation_path),
    )


def parse_annotation(relative_image_path: Path) -> ImageAnnotation:
    """Parse the size, breed, body box, and head box for one image."""
    annotation_path = ANNOTATION_ROOT / f"{relative_image_path}.xml"
    root = ET.parse(annotation_path).getroot()

    size_element = require_element(root, "size", annotation_path)

    image_width = require_integer(size_element, "width", annotation_path)
    image_height = require_integer(size_element, "height", annotation_path)

    dog_elements = root.findall("object")

    if not dog_elements:
        raise ValueError(f"No dog objects in annotation: {annotation_path}")

    dogs: list[AnnotatedDog] = []

    for dog_element in dog_elements:
        breed = dog_element.findtext("name")

        if breed is None:
            raise ValueError(f"Missing breed name in annotation: {annotation_path}")

        body_box_element = require_element(
            dog_element,
            "bodybndbox",
            annotation_path,
        )

        head_box_element = require_element(
            dog_element,
            "headbndbox",
            annotation_path,
        )

        dogs.append(
            AnnotatedDog(
                breed=breed,
                body_box=parse_box(body_box_element, annotation_path),
                head_box=parse_box(head_box_element, annotation_path),
            )
        )

    return ImageAnnotation(
        width=image_width,
        height=image_height,
        dogs=tuple(dogs),
    )


def inspect_sample_annotation(relative_image_path: Path) -> None:
    """Print one parsed annotation as a quick format sanity check."""
    annotation = parse_annotation(relative_image_path)
    first_dog = annotation.dogs[0]

    print(f"Sample image: {relative_image_path}")
    print(f"Image size: {annotation.width} x {annotation.height}")
    print(f"Annotated dogs: {len(annotation.dogs)}")
    print(f"First breed: {first_dog.breed}")
    print(f"First body box: {first_dog.body_box}")
    print(f"First head box: {first_dog.head_box}")


def is_valid_box(
    box: BoundingBox,
    image_width: int,
    image_height: int,
) -> bool:
    """Check that a box has area and lies within the supplied dimensions."""
    return (
        0 <= box.xmin < box.xmax <= image_width
        and 0 <= box.ymin < box.ymax <= image_height
    )


def clamp_box(
    box: BoundingBox,
    image_width: int,
    image_height: int,
) -> BoundingBox:
    """Restrict all box coordinates to the supplied image boundaries."""
    return BoundingBox(
        xmin=min(max(box.xmin, 0), image_width),
        ymin=min(max(box.ymin, 0), image_height),
        xmax=min(max(box.xmax, 0), image_width),
        ymax=min(max(box.ymax, 0), image_height),
    )


def percentile(sorted_values: list[int], fraction: float) -> int:
    """Select a nearest-rank value from an already sorted integer list."""
    if not sorted_values:
        raise ValueError("Cannot calculate a percentile without values")

    index = round((len(sorted_values) - 1) * fraction)
    return sorted_values[index]


def inspect_dataset(
    train_paths: list[Path],
    validation_paths: list[Path],
) -> None:
    """Audit annotation validity, crop sizes, and class balance in both splits."""
    object_counts: Counter[int] = Counter()
    train_breed_counts: Counter[str] = Counter()
    validation_breed_counts: Counter[str] = Counter()

    body_minimum_sides: list[int] = []

    invalid_image_sizes = 0
    invalid_body_boxes = 0
    invalid_head_boxes = 0
    body_crops_below_224 = 0
    total_dogs = 0
    recoverable_body_boxes = 0
    unrecoverable_body_boxes = 0

    body_box_adjustments: list[int] = []

    invalid_body_examples: list[tuple[Path, BoundingBox, BoundingBox]] = []

    # This diagnostic uses XML-declared dimensions. Some low-resolution XML
    # files have width and height swapped, so the training loader must validate
    # boxes against the dimensions of the decoded JPEG instead.
    splits = (
        ("training", train_paths, train_breed_counts),
        ("validation", validation_paths, validation_breed_counts),
    )

    for split_name, image_paths, breed_counts in splits:
        for image_index, relative_image_path in enumerate(
            image_paths,
            start=1,
        ):
            annotation = parse_annotation(relative_image_path)

            if annotation.width <= 0 or annotation.height <= 0:
                invalid_image_sizes += 1

            object_counts[len(annotation.dogs)] += 1

            for dog in annotation.dogs:
                total_dogs += 1
                breed_counts[dog.breed] += 1

                if is_valid_box(
                    dog.body_box,
                    annotation.width,
                    annotation.height,
                ):
                    minimum_side = min(
                        dog.body_box.width,
                        dog.body_box.height,
                    )

                    body_minimum_sides.append(minimum_side)

                    if minimum_side < 224:
                        body_crops_below_224 += 1
                else:
                    invalid_body_boxes += 1

                    clamped_body_box = clamp_box(
                        dog.body_box,
                        annotation.width,
                        annotation.height,
                    )

                    maximum_adjustment = max(
                        abs(dog.body_box.xmin - clamped_body_box.xmin),
                        abs(dog.body_box.ymin - clamped_body_box.ymin),
                        abs(dog.body_box.xmax - clamped_body_box.xmax),
                        abs(dog.body_box.ymax - clamped_body_box.ymax),
                    )

                    body_box_adjustments.append(maximum_adjustment)

                    if is_valid_box(
                        clamped_body_box,
                        annotation.width,
                        annotation.height,
                    ):
                        recoverable_body_boxes += 1

                        minimum_side = min(
                            clamped_body_box.width,
                            clamped_body_box.height,
                        )

                        body_minimum_sides.append(minimum_side)

                        if minimum_side < 224:
                            body_crops_below_224 += 1
                    else:
                        unrecoverable_body_boxes += 1

                    if len(invalid_body_examples) < 5:
                        invalid_body_examples.append(
                            (
                                relative_image_path,
                                dog.body_box,
                                clamped_body_box,
                            )
                        )

                if not is_valid_box(
                    dog.head_box,
                    annotation.width,
                    annotation.height,
                ):
                    invalid_head_boxes += 1

            if image_index % 10_000 == 0:
                print(f"Parsed {image_index:,} " f"{split_name} annotations...")

    body_minimum_sides.sort()
    body_box_adjustments.sort()

    print()
    print(f"Total annotated dogs: {total_dogs:,}")
    print(f"Invalid image sizes: {invalid_image_sizes:,}")
    print(f"Invalid body boxes: {invalid_body_boxes:,}")
    print(f"Invalid head boxes: {invalid_head_boxes:,}")
    print(f"Recoverable by clamping: {recoverable_body_boxes:,}")
    print(f"Unrecoverable after clamping: {unrecoverable_body_boxes:,}")

    if body_box_adjustments:
        print("Required body-box adjustment:")
        print(f"  p50: {percentile(body_box_adjustments, 0.50):,} px")
        print(f"  p90: {percentile(body_box_adjustments, 0.90):,} px")
        print(f"  maximum: {body_box_adjustments[-1]:,} px")

    print("Invalid body-box examples:")

    for image_path, original_box, clamped_box in invalid_body_examples:
        print(f"  {image_path}")
        print(f"    original: {original_box}")
        print(f"    clamped:  {clamped_box}")

    print("Images grouped by annotated dog count:")

    for dog_count, image_count in sorted(object_counts.items()):
        print(f"  {dog_count} dog(s): {image_count:,} images")

    print()
    print("Body crop minimum-side distribution:")
    print(f"  p10: {percentile(body_minimum_sides, 0.10):,} px")
    print(f"  p50: {percentile(body_minimum_sides, 0.50):,} px")
    print(f"  p90: {percentile(body_minimum_sides, 0.90):,} px")

    below_224_percentage = body_crops_below_224 / len(body_minimum_sides) * 100

    print(f"  Below 224 px: {body_crops_below_224:,} " f"({below_224_percentage:.2f}%)")

    print()
    print(f"Training breeds: {len(train_breed_counts):,}")
    print(
        "Training examples per breed: "
        f"{min(train_breed_counts.values()):,}-"
        f"{max(train_breed_counts.values()):,}"
    )

    print(f"Validation breeds: {len(validation_breed_counts):,}")
    print(
        "Validation examples per breed: "
        f"{min(validation_breed_counts.values()):,}-"
        f"{max(validation_breed_counts.values()):,}"
    )

    same_breeds = set(train_breed_counts) == set(validation_breed_counts)

    print(f"Both splits contain the same breeds: {same_breeds}")


def main() -> None:
    """Load the official splits and print the complete dataset audit."""
    train_paths = load_split_paths(TRAIN_SPLIT_PATH)
    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    print(f"Training images: {len(train_paths):,}")
    print(f"Validation images: {len(validation_paths):,}")
    print(f"Total images: {len(train_paths) + len(validation_paths):,}")
    print()
    inspect_sample_annotation(train_paths[0])
    print()
    inspect_dataset(train_paths, validation_paths)


if __name__ == "__main__":
    main()
