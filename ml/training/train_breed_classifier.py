from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets.tsinghua_dogs import (
    TRAINING_AUGMENTATION,
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_breed_mapping,
    load_split_paths,
)

from training.breed_classifier import build_breed_classifier

BATCH_SIZE = 4
MAX_DEBUG_BATCHES = 3
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_batches: int | None = None,
) -> float:
    """Train the model for one pass over the supplied batches."""

    model.train()

    # Keep the frozen backbone's BatchNorm statistics unchanged.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0
    processed_examples = 0

    for batch_number, (batch_images, batch_breed_ids) in enumerate(
        data_loader,
        start=1,
    ):
        if max_batches is not None and batch_number > max_batches:
            break

        # 1. Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        # 2. Forward pass: produce one breed score per class and image.
        output_logits = model(batch_images)

        # 3. Compare the predicted scores with the correct breed IDs.
        loss = loss_function(output_logits, batch_breed_ids)

        # 4. Backward pass: calculate gradients for trainable parameters.
        loss.backward()

        # 5. Update the trainable parameters using those gradients.
        optimizer.step()

        batch_size = batch_images.size(0)
        total_loss += loss.item() * batch_size
        processed_examples += batch_size

        print(f"Batch {batch_number}: loss={loss.item():.4f}")

    if processed_examples == 0:
        raise ValueError("Training did not receive any examples")

    return total_loss / processed_examples


def evaluate(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Calculate average loss and top-one accuracy without training."""

    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    processed_examples = 0

    with torch.inference_mode():
        for batch_number, (batch_images, batch_breed_ids) in enumerate(
            data_loader,
            start=1,
        ):
            if max_batches is not None and batch_number > max_batches:
                break

            output_logits = model(batch_images)
            loss = loss_function(output_logits, batch_breed_ids)

            predicted_breed_ids = output_logits.argmax(dim=1)

            batch_size = batch_images.size(0)
            total_loss += loss.item() * batch_size
            processed_examples += batch_size

            correct_predictions += (predicted_breed_ids == batch_breed_ids).sum().item()

    if processed_examples == 0:
        raise ValueError("Validation did not receive any examples")

    average_loss = total_loss / processed_examples
    accuracy = correct_predictions / processed_examples

    return average_loss, accuracy


def main() -> None:
    """Build the training pipeline and run a short local smoke test."""

    model = build_breed_classifier()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_paths = load_split_paths(TRAIN_SPLIT_PATH)
    _, breed_to_id = build_breed_mapping(train_paths)

    train_dataset = TsinghuaDogsDataset(
        train_paths,
        breed_to_id,
        image_augmentation=TRAINING_AUGMENTATION,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    loss_function = nn.CrossEntropyLoss()

    average_training_loss = train_one_epoch(
        model=model,
        data_loader=train_data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        max_batches=MAX_DEBUG_BATCHES,
    )

    # Validation
    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    validation_loss, validation_accuracy = evaluate(
        model=model,
        data_loader=validation_data_loader,
        loss_function=loss_function,
        max_batches=MAX_DEBUG_BATCHES,
    )

    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.2%}")


if __name__ == "__main__":
    main()
