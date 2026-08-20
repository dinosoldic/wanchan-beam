"""Evaluate a saved dog-breed classifier checkpoint."""

import argparse
from collections.abc import Iterable
from pathlib import Path
from dataclasses import dataclass

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


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and predictions collected over the complete validation set."""

    loss: float
    top_one_accuracy: float
    top_two_accuracy: float
    true_breed_ids: tuple[int, ...]
    predicted_breed_ids: tuple[int, ...]


@dataclass(frozen=True)
class BreedAccuracy:
    """Top-1 validation accuracy for one breed."""

    breed_id: int
    breed_name: str
    correct_predictions: int
    total_examples: int

    @property
    def accuracy(self) -> float:
        return self.correct_predictions / self.total_examples


@dataclass(frozen=True)
class BreedConfusion:
    """One incorrect true-breed and predicted-breed combination."""

    true_breed_name: str
    predicted_breed_name: str
    incorrect_predictions: int
    true_breed_examples: int

    @property
    def error_rate(self) -> float:
        return self.incorrect_predictions / self.true_breed_examples


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
) -> EvaluationResult:
    """Calculate validation loss and top-one/top-two accuracy."""

    loss_function = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct_top_one = 0
    correct_top_two = 0
    processed_examples = 0
    true_breed_ids: list[int] = []
    predicted_breed_ids: list[int] = []

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

            predicted_top_one_ids = top_two_breed_ids[:, 0]
            true_breed_ids.extend(
                int(breed_id) for breed_id in batch_breed_ids.cpu().tolist()
            )
            predicted_breed_ids.extend(
                int(breed_id) for breed_id in predicted_top_one_ids.cpu().tolist()
            )

            correct_top_one += (predicted_top_one_ids == batch_breed_ids).sum().item()

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

    return EvaluationResult(
        loss=total_loss / processed_examples,
        top_one_accuracy=correct_top_one / processed_examples,
        top_two_accuracy=correct_top_two / processed_examples,
        true_breed_ids=tuple(true_breed_ids),
        predicted_breed_ids=tuple(predicted_breed_ids),
    )


def build_confusion_matrix(
    evaluation_result: EvaluationResult,
    number_of_breeds: int,
) -> Tensor:
    """Count every true-breed and predicted-breed combination."""

    if len(evaluation_result.true_breed_ids) != len(
        evaluation_result.predicted_breed_ids
    ):
        raise ValueError("True and predicted breed ID counts do not match")

    confusion_matrix = torch.zeros(
        (number_of_breeds, number_of_breeds),
        dtype=torch.int64,
    )

    for true_breed_id, predicted_breed_id in zip(
        evaluation_result.true_breed_ids,
        evaluation_result.predicted_breed_ids,
        strict=True,
    ):
        confusion_matrix[true_breed_id, predicted_breed_id] += 1

    return confusion_matrix


def calculate_breed_accuracies(
    confusion_matrix: Tensor,
    breed_names: tuple[str, ...],
) -> tuple[BreedAccuracy, ...]:
    """Calculate top-1 accuracy independently for every breed."""

    if tuple(confusion_matrix.shape) != (len(breed_names), len(breed_names)):
        raise ValueError("Confusion matrix shape does not match the breed count")

    breed_accuracies: list[BreedAccuracy] = []

    for breed_id, breed_name in enumerate(breed_names):
        total_examples = int(confusion_matrix[breed_id].sum().item())
        correct_predictions = int(confusion_matrix[breed_id, breed_id].item())

        if total_examples == 0:
            raise ValueError(f"Breed has no validation examples: {breed_name}")

        breed_accuracies.append(
            BreedAccuracy(
                breed_id=breed_id,
                breed_name=breed_name,
                correct_predictions=correct_predictions,
                total_examples=total_examples,
            )
        )

    return tuple(breed_accuracies)


def find_common_confusions(
    confusion_matrix: Tensor,
    breed_names: tuple[str, ...],
) -> tuple[BreedConfusion, ...]:
    """Return incorrect breed pairs ordered by their frequency."""

    confusions: list[BreedConfusion] = []

    for true_breed_id, true_breed_name in enumerate(breed_names):
        true_breed_examples = int(confusion_matrix[true_breed_id].sum().item())

        for predicted_breed_id, predicted_breed_name in enumerate(breed_names):
            if predicted_breed_id == true_breed_id:
                continue

            incorrect_predictions = int(
                confusion_matrix[true_breed_id, predicted_breed_id].item()
            )

            if incorrect_predictions == 0:
                continue

            confusions.append(
                BreedConfusion(
                    true_breed_name=true_breed_name,
                    predicted_breed_name=predicted_breed_name,
                    incorrect_predictions=incorrect_predictions,
                    true_breed_examples=true_breed_examples,
                )
            )

    return tuple(
        sorted(
            confusions,
            key=lambda confusion: confusion.incorrect_predictions,
            reverse=True,
        )
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

    evaluation_result = evaluate_model(
        model=model,
        data_loader=validation_data_loader,
        device=device,
    )

    confusion_matrix = build_confusion_matrix(
        evaluation_result=evaluation_result,
        number_of_breeds=len(breed_names),
    )

    breed_accuracies = calculate_breed_accuracies(
        confusion_matrix=confusion_matrix,
        breed_names=breed_names,
    )

    lowest_breed_accuracies = sorted(
        breed_accuracies,
        key=lambda breed_result: breed_result.accuracy,
    )[:10]

    common_confusions = find_common_confusions(
        confusion_matrix=confusion_matrix,
        breed_names=breed_names,
    )

    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Evaluation device: {device}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Saved validation loss: {checkpoint['validation_loss']:.4f}")
    print("Saved top-1 accuracy: " f"{checkpoint['validation_accuracy']:.2%}")

    print()
    print(f"Recalculated validation loss: {evaluation_result.loss:.4f}")
    print(f"Recalculated top-1 accuracy: {evaluation_result.top_one_accuracy:.2%}")
    print(f"Top-2 accuracy: {evaluation_result.top_two_accuracy:.2%}")

    print()
    print(f"Confusion matrix shape: {tuple(confusion_matrix.shape)}")
    print(f"Predictions in matrix: {confusion_matrix.sum().item():,}")
    print(
        f"Correct predictions in matrix: {confusion_matrix.diagonal().sum().item():,}"
    )

    print()
    print("Lowest top-1 breed accuracies:")

    for breed_result in lowest_breed_accuracies:
        print(
            f"  {breed_result.breed_name}: "
            f"{breed_result.accuracy:.2%} "
            f"({breed_result.correct_predictions}/{breed_result.total_examples})"
        )

    print()
    print("Most common breed confusions:")
    for confusion in common_confusions[:15]:
        print(
            f"  {confusion.true_breed_name} -> "
            f"{confusion.predicted_breed_name}: "
            f"{confusion.incorrect_predictions}/"
            f"{confusion.true_breed_examples} "
            f"({confusion.error_rate:.2%})"
        )


if __name__ == "__main__":
    main()
