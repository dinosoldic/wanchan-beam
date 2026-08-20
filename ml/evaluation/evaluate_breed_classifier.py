"""Evaluate a saved dog-breed classifier checkpoint."""

import argparse
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    build_breed_mapping,
    load_split_paths,
    TsinghuaDogsDataset,
    VALIDATION_SPLIT_PATH,
)
from training.breed_classifier import build_breed_classifier

EVALUATION_BATCH_SIZE = 64
NUM_WORKERS = 4


def parse_checkpoint_path() -> Path:
    """Read and validate the checkpoint path supplied on the command line."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained dog-breed classifier checkpoint."
    )
    parser.add_argument(
        "checkpoint_path",
        type=Path,
        help="Path to the .pt checkpoint to evaluate",
    )
    arguments = parser.parse_args()
    checkpoint_path: Path = arguments.checkpoint_path

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def select_device() -> torch.device:
    """Use CUDA when available, otherwise evaluate on the CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_checkpoint(
    checkpoint_path: Path,
    expected_breed_names: tuple[str, ...],
    device: torch.device,
) -> tuple[nn.Module, dict]:
    """Load a complete model checkpoint and verify its breed ordering."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    checkpoint_breed_names = tuple(checkpoint["breed_names"])

    if checkpoint_breed_names != expected_breed_names:
        raise ValueError("Checkpoint breed ordering does not match the current dataset")

    model = build_breed_classifier(
        number_of_breeds=len(expected_breed_names),
        use_pretrained_weights=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> tuple[float, float, float]:
    """Calculate validation loss and top-one/top-two accuracy."""

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_top_one = 0
    correct_top_two = 0
    processed_examples = 0

    model.eval()

    with torch.inference_mode():
        for batch_images, batch_breed_ids in data_loader:
            batch_images = batch_images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            batch_breed_ids = batch_breed_ids.to(
                device,
                non_blocking=device.type == "cuda",
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                output_logits = model(batch_images)
                loss = loss_function(output_logits, batch_breed_ids)

            top_two_breed_ids = output_logits.topk(
                k=2,
                dim=1,
            ).indices

            correct_top_one += (top_two_breed_ids[:, 0] == batch_breed_ids).sum().item()

            correct_top_two += (
                (top_two_breed_ids == batch_breed_ids.unsqueeze(1))
                .any(dim=1)
                .sum()
                .item()
            )

            batch_size = batch_images.size(0)
            total_loss += loss.item() * batch_size
            processed_examples += batch_size

    if processed_examples == 0:
        raise ValueError("Evaluation did not receive any examples")

    return (
        total_loss / processed_examples,
        correct_top_one / processed_examples,
        correct_top_two / processed_examples,
    )


def main() -> None:
    """Load and evaluate the classifier checkpoint selected by the user."""

    checkpoint_path = parse_checkpoint_path()
    device = select_device()

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    model, checkpoint = load_model_checkpoint(
        checkpoint_path=checkpoint_path,
        expected_breed_names=breed_names,
        device=device,
    )

    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=EVALUATION_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    validation_loss, top_one_accuracy, top_two_accuracy = evaluate_model(
        model=model,
        data_loader=validation_data_loader,
        device=device,
    )

    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Evaluation device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Saved validation loss: {checkpoint['validation_loss']:.4f}")
    print("Saved top-1 accuracy: " f"{checkpoint['validation_accuracy']:.2%}")

    print()
    print(f"Recalculated validation loss: {validation_loss:.4f}")
    print(f"Recalculated top-1 accuracy: {top_one_accuracy:.2%}")
    print(f"Top-2 accuracy: {top_two_accuracy:.2%}")


if __name__ == "__main__":
    main()
