from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_training_augmentation,
    build_breed_mapping,
    get_breed_name,
    load_split_paths,
)

from training.breed_classifier import build_breed_classifier

### constants
CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "artifacts" / "classifier" / "checkpoints"
)

DEBUG_MODE = True
RESUME_FROM_CHECKPOINT = True

BATCH_SIZE = 4 if DEBUG_MODE else 64
NUM_WORKERS = 0 if DEBUG_MODE else 4
MAX_BATCHES = 3 if DEBUG_MODE else None
TOTAL_EPOCHS = 3 if DEBUG_MODE else 10
RANDOM_SEED = 42
BATCH_LOG_INTERVAL = 1 if DEBUG_MODE else 50

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

CLASSIFIER_HIDDEN_FEATURES: int | None = 512
DROPOUT_PROBABILITY = 0.4
HEAD_EXPERIMENT_NAME = "head-512"

RUN_NAME = f"{HEAD_EXPERIMENT_NAME}-debug" if DEBUG_MODE else HEAD_EXPERIMENT_NAME


### funcs
def select_device() -> torch.device:
    """Use an NVIDIA CUDA GPU when available, otherwise use the CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seed(seed: int) -> None:
    """Initialize PyTorch randomness consistently for a fresh training run."""

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_balanced_sample_weights(image_paths: list[Path]) -> list[float]:
    """Give images from rare breeds a greater sampling probability."""

    breed_counts = Counter(get_breed_name(image_path) for image_path in image_paths)

    return [
        1.0 / breed_counts[get_breed_name(image_path)] for image_path in image_paths
    ]


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None = None,
    log_interval: int = 1,
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

        # pass to device (GPU or CPU)
        batch_images = batch_images.to(device, non_blocking=device.type == "cuda")
        batch_breed_ids = batch_breed_ids.to(device, non_blocking=device.type == "cuda")

        # 1. Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        with torch.amp.autocast(
            device_type=device.type,
            enabled=gradient_scaler.is_enabled(),
        ):
            # 2. Forward pass: produce one breed score per class and image.
            output_logits = model(batch_images)

            # 3. Compare the predicted scores with the correct breed IDs.
            loss = loss_function(output_logits, batch_breed_ids)

        # 4. Scale the loss and calculate gradients.
        gradient_scaler.scale(loss).backward()

        # 5. Unscale gradients and update the trainable parameters.
        gradient_scaler.step(optimizer)

        # Adjust the scale for the next batch.
        gradient_scaler.update()

        batch_size = batch_images.size(0)
        total_loss += loss.item() * batch_size
        processed_examples += batch_size

        if batch_number == 1 or batch_number % log_interval == 0:
            print(f"Batch {batch_number}: loss={loss.item():.4f}")

    if processed_examples == 0:
        raise ValueError("Training did not receive any examples")

    return total_loss / processed_examples


def evaluate(
    model: nn.Module,
    data_loader: Iterable[tuple[Tensor, Tensor]],
    loss_function: nn.Module,
    device: torch.device,
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

            # pass to device (GPU or CPU)
            batch_images = batch_images.to(device, non_blocking=device.type == "cuda")
            batch_breed_ids = batch_breed_ids.to(
                device, non_blocking=device.type == "cuda"
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
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


def save_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    epoch_number: int,
    breed_names: tuple[str, ...],
    validation_loss: float,
    validation_accuracy: float,
    best_validation_loss: float,
    learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    sampler_generator: torch.Generator | None = None,
    early_stopping_counter: int = 0,
) -> None:
    """Save enough training state to resume from the current epoch."""

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch_number,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "gradient_scaler_state_dict": gradient_scaler.state_dict(),
        "breed_names": breed_names,
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "best_validation_loss": best_validation_loss,
        "learning_rate_scheduler_state_dict": (
            learning_rate_scheduler.state_dict()
            if learning_rate_scheduler is not None
            else None
        ),
        "sampler_generator_state": (
            sampler_generator.get_state() if sampler_generator is not None else None
        ),
        "early_stopping_counter": early_stopping_counter,
    }

    torch.save(checkpoint, checkpoint_path)


def load_training_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    gradient_scaler: torch.amp.GradScaler,
    expected_breed_names: tuple[str, ...],
    device: torch.device,
    learning_rate_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    sampler_generator: torch.Generator | None = None,
) -> tuple[int, float, int]:
    """Restore training state and return its epoch, best loss, and stop counter."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    checkpoint_breed_names = tuple(checkpoint["breed_names"])

    if checkpoint_breed_names != expected_breed_names:
        raise ValueError("Checkpoint breed ordering does not match the current dataset")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler_state = checkpoint["gradient_scaler_state_dict"]

    # CPU checkpoints contain no active CUDA scaler state.
    if gradient_scaler.is_enabled() and scaler_state:
        gradient_scaler.load_state_dict(scaler_state)

    # restore LR scheduler
    scheduler_state = checkpoint.get("learning_rate_scheduler_state_dict")
    if learning_rate_scheduler is not None and scheduler_state:
        learning_rate_scheduler.load_state_dict(scheduler_state)

    sampler_generator_state = checkpoint.get("sampler_generator_state")
    if sampler_generator is not None and sampler_generator_state is not None:
        # Checkpoints loaded onto CUDA may also move this byte tensor, but a
        # CPU-based sampler generator requires its state to remain on the CPU.
        sampler_generator.set_state(sampler_generator_state.cpu())

    best_validation_loss = float(
        checkpoint.get(
            "best_validation_loss",
            checkpoint["validation_loss"],
        )
    )

    early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))

    return int(checkpoint["epoch"]), best_validation_loss, early_stopping_counter


def main() -> None:
    """Build the training pipeline and run a short local smoke test."""

    set_random_seed(RANDOM_SEED)

    device = select_device()
    if not DEBUG_MODE and device.type != "cuda":
        raise RuntimeError(
            "Full training mode requires CUDA. Enable DEBUG_MODE for CPU testing."
        )

    model = build_breed_classifier(
        use_pretrained_weights=True,
        dropout_probability=DROPOUT_PROBABILITY,
        classifier_hidden_features=CLASSIFIER_HIDDEN_FEATURES,
    ).to(device)

    mixed_precision_enabled = device.type == "cuda"

    gradient_scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=mixed_precision_enabled,
    )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    train_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(train_paths)

    training_augmentation = build_training_augmentation(
        use_geometric_augmentation=False,
    )

    train_dataset = TsinghuaDogsDataset(
        train_paths,
        breed_to_id,
        image_augmentation=training_augmentation,
    )

    # Use a fixed random seed for the sampler to ensure reproducibility across runs.
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(RANDOM_SEED)

    # balance the training dataset by giving rare breeds a higher sampling probability
    train_sample_weights = build_balanced_sample_weights(train_paths)
    train_sampler = WeightedRandomSampler(
        weights=train_sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=sampler_generator,
    )

    # train set
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    loss_function = nn.CrossEntropyLoss()

    # Validation uses the official split without random augmentation
    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    # check if checkpoint exists and load
    latest_checkpoint_path = CHECKPOINT_DIRECTORY / f"{RUN_NAME}-latest.pt"
    best_checkpoint_path = CHECKPOINT_DIRECTORY / f"{RUN_NAME}-best.pt"

    starting_epoch = 1
    best_validation_loss = float("inf")
    checkpoint_status = "No training checkpoint found; starting a fresh run"

    if RESUME_FROM_CHECKPOINT and latest_checkpoint_path.exists():
        completed_epoch, best_validation_loss, _ = load_training_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            expected_breed_names=breed_names,
            device=device,
            sampler_generator=sampler_generator,
        )

        starting_epoch = completed_epoch + 1
        checkpoint_status = f"Resuming after epoch {completed_epoch}"

    device_description = device.type
    if device.type == "cuda":
        device_description += f" ({torch.cuda.get_device_name(device)})"

    print(f"Training mode: {'debug' if DEBUG_MODE else 'full'}")
    print(f"Training device: {device_description}")
    print(
        f"Data: {len(train_dataset):,} training samples "
        f"({len(train_data_loader):,} batches), "
        f"{len(validation_dataset):,} validation samples "
        f"({len(validation_data_loader):,} batches)"
    )
    print(f"Optimizer: learning rate={LEARNING_RATE}, weight decay={WEIGHT_DECAY}")
    print("Geometric augmentation: False")
    print(f"Mixed precision: {mixed_precision_enabled}")
    print(f"Checkpoint: {checkpoint_status}")

    print(f"Head experiment: {HEAD_EXPERIMENT_NAME}")
    print(f"Classifier hidden features: {CLASSIFIER_HIDDEN_FEATURES}")

    # training loop
    for epoch_number in range(starting_epoch, TOTAL_EPOCHS + 1):
        print()
        print(f"Epoch {epoch_number}/{TOTAL_EPOCHS}")

        average_training_loss = train_one_epoch(
            model=model,
            data_loader=train_data_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            device=device,
            max_batches=MAX_BATCHES,
            log_interval=BATCH_LOG_INTERVAL,
        )

        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_data_loader,
            loss_function=loss_function,
            device=device,
            max_batches=MAX_BATCHES,
        )

        print(f"Training loss: {average_training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.2%}")

        # Keep a separate copy of the model with the lowest validation loss.
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss

            save_training_checkpoint(
                checkpoint_path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                gradient_scaler=gradient_scaler,
                epoch_number=epoch_number,
                breed_names=breed_names,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                best_validation_loss=best_validation_loss,
                sampler_generator=sampler_generator,
            )

            print(f"Saved new best checkpoint: {best_checkpoint_path}")

        # save current epoch checkpoint
        save_training_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            epoch_number=epoch_number,
            breed_names=breed_names,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            best_validation_loss=best_validation_loss,
            sampler_generator=sampler_generator,
        )

        print(f"Saved checkpoint: {latest_checkpoint_path}")


if __name__ == "__main__":
    main()
