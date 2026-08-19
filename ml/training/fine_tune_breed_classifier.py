"""Partially fine-tune the trained MobileNetV3 breed classifier."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.tsinghua_dogs import (
    TRAINING_AUGMENTATION,
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_breed_mapping,
    load_split_paths,
)
from training.breed_classifier import (
    build_breed_classifier,
    unfreeze_for_fine_tuning,
)
from training.train_breed_classifier import (
    build_balanced_sample_weights,
    evaluate,
    load_training_checkpoint,
    save_training_checkpoint,
    set_random_seed,
    train_one_epoch,
)

### consts
ML_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIRECTORY = ML_ROOT / "artifacts" / "classifier" / "checkpoints"

EXPERIMENT_NAME = "2modules"

BASELINE_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "head-baseline-best.pt"
FINE_TUNE_LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / f"fine-tune-{EXPERIMENT_NAME}-latest.pt"
)
FINE_TUNE_BEST_LOSS_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / f"fine-tune-{EXPERIMENT_NAME}-best-loss.pt"
)

FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH = (
    CHECKPOINT_DIRECTORY / f"fine-tune-{EXPERIMENT_NAME}-best-accuracy.pt"
)

NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE = 2

BATCH_SIZE = 64
NUMBER_OF_WORKERS = 4
RANDOM_SEED = 42
TOTAL_EPOCHS = 5
BATCH_LOG_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 2
RESUME_FROM_CHECKPOINT = True

# The classifier can adapt faster, while the pretrained feature extractor uses a
# smaller learning rate to avoid destroying useful ImageNet features.
CLASSIFIER_LEARNING_RATE = 0.0001
FEATURE_LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.0001


### funcs
def select_device() -> torch.device:
    """Use CUDA when available, otherwise use the CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_trainable_parameters(module: nn.Module) -> int:
    """Count parameters that will receive gradient updates."""

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    """Fine-tune the deepest MobileNetV3 modules and classifier."""

    set_random_seed(RANDOM_SEED)

    device = select_device()

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    training_dataset = TsinghuaDogsDataset(
        training_paths,
        breed_to_id,
        image_augmentation=TRAINING_AUGMENTATION,
    )

    # A dedicated generator makes the balanced sampling order reproducible.
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(RANDOM_SEED)

    training_sample_weights = build_balanced_sample_weights(training_paths)

    training_sampler = WeightedRandomSampler(
        weights=training_sample_weights,
        num_samples=len(training_dataset),
        replacement=True,
        generator=sampler_generator,
    )

    training_data_loader = DataLoader(
        training_dataset,
        batch_size=BATCH_SIZE,
        sampler=training_sampler,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUMBER_OF_WORKERS > 0,
    )

    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)

    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUMBER_OF_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUMBER_OF_WORKERS > 0,
    )

    baseline_checkpoint = torch.load(
        BASELINE_CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    # Class IDs depend on this exact ordering, so a mismatch would train each
    # output neuron against the wrong breed.
    if tuple(baseline_checkpoint["breed_names"]) != breed_names:
        raise ValueError("Checkpoint breed ordering does not match the current dataset")

    model = build_breed_classifier(
        number_of_breeds=len(breed_names),
        use_pretrained_weights=False,
    ).to(device)

    model.load_state_dict(baseline_checkpoint["model_state_dict"])

    # The factory initially freezes the feature extractor. We then reopen only
    # its deepest modules, which contain the most task-specific visual features.
    unfreeze_for_fine_tuning(
        model,
        number_of_feature_modules=NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE,
    )

    trainable_parameter_count = count_trainable_parameters(model)
    feature_modules = list(model.features.children())

    unfrozen_feature_modules = feature_modules[-NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE:]

    # Separate parameter groups let AdamW update the classifier and pretrained
    # feature modules at different learning rates.
    classifier_parameter_group = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]

    feature_parameter_group = [
        parameter
        for feature_module in unfrozen_feature_modules
        for parameter in feature_module.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": classifier_parameter_group,
                "lr": CLASSIFIER_LEARNING_RATE,
            },
            {
                "params": feature_parameter_group,
                "lr": FEATURE_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.CrossEntropyLoss()

    # Mixed precision reduces GPU memory use and generally speeds up CUDA training.
    mixed_precision_enabled = device.type == "cuda"

    gradient_scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=mixed_precision_enabled,
    )

    starting_epoch = 1
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_accuracy_improvement = 0

    if RESUME_FROM_CHECKPOINT and FINE_TUNE_LATEST_CHECKPOINT_PATH.exists():
        completed_epoch, best_validation_loss = load_training_checkpoint(
            checkpoint_path=FINE_TUNE_LATEST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            expected_breed_names=breed_names,
            device=device,
        )

        starting_epoch = completed_epoch + 1

        if FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH.exists():
            best_accuracy_checkpoint = torch.load(
                FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH,
                map_location=device,
                weights_only=True,
            )

            if tuple(best_accuracy_checkpoint["breed_names"]) != breed_names:
                raise ValueError(
                    "Best-accuracy checkpoint breed ordering does not "
                    "match the current dataset"
                )

            best_validation_accuracy = float(
                best_accuracy_checkpoint["validation_accuracy"]
            )

        print(f"Resuming fine-tuning after epoch {completed_epoch}")
    else:
        print("No fine-tuning checkpoint found; " "starting from the baseline model")

    # Catch accidentally trainable parameters that were omitted from the
    # optimizer. Such parameters would receive gradients but never be updated.
    optimized_parameter_count = sum(
        parameter.numel()
        for parameter in (classifier_parameter_group + feature_parameter_group)
    )

    if optimized_parameter_count != trainable_parameter_count:
        raise RuntimeError("Optimizer parameters do not match all trainable parameters")

    device_description = device.type
    if device.type == "cuda":
        device_description += f" ({torch.cuda.get_device_name(device)})"

    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Fine-tuning device: {device_description}")
    print(
        f"Baseline: epoch {baseline_checkpoint['epoch']}, "
        f"validation loss={baseline_checkpoint['validation_loss']:.4f}, "
        f"accuracy={baseline_checkpoint['validation_accuracy']:.2%}"
    )
    print(
        f"Model: {NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE} feature modules unfrozen, "
        f"{trainable_parameter_count:,} trainable parameters"
    )
    print(
        f"Optimizer: classifier lr={CLASSIFIER_LEARNING_RATE}, "
        f"feature lr={FEATURE_LEARNING_RATE}, weight decay={WEIGHT_DECAY}"
    )
    print(
        f"Data: {len(training_dataset):,} training samples "
        f"({len(training_data_loader):,} batches), "
        f"{len(validation_dataset):,} validation samples "
        f"({len(validation_data_loader):,} batches)"
    )
    print(f"Mixed precision: {mixed_precision_enabled}")

    for epoch_number in range(starting_epoch, TOTAL_EPOCHS + 1):
        print()
        print(f"Epoch {epoch_number}/{TOTAL_EPOCHS}")

        # Train on the complete balanced training set.
        average_training_loss = train_one_epoch(
            model=model,
            data_loader=training_data_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            device=device,
            log_interval=BATCH_LOG_INTERVAL,
        )

        # Measure generalization without updating any parameters.
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_data_loader,
            loss_function=loss_function,
            device=device,
        )

        print(f"Training loss: {average_training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.2%}")

        validation_improved = validation_loss < best_validation_loss

        if validation_improved:
            best_validation_loss = validation_loss

            save_training_checkpoint(
                checkpoint_path=FINE_TUNE_BEST_LOSS_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                gradient_scaler=gradient_scaler,
                epoch_number=epoch_number,
                breed_names=breed_names,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                best_validation_loss=best_validation_loss,
            )

            print(
                "Saved new best-loss checkpoint: "
                f"{FINE_TUNE_BEST_LOSS_CHECKPOINT_PATH}"
            )

        # Accuracy and loss can favor different epochs, so preserve both.
        accuracy_improved = validation_accuracy > best_validation_accuracy

        if accuracy_improved:
            best_validation_accuracy = validation_accuracy
            epochs_without_accuracy_improvement = 0

            save_training_checkpoint(
                checkpoint_path=FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                gradient_scaler=gradient_scaler,
                epoch_number=epoch_number,
                breed_names=breed_names,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                best_validation_loss=best_validation_loss,
            )

            print(
                "Saved new best-accuracy checkpoint: "
                f"{FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH}"
            )
        else:
            epochs_without_accuracy_improvement += 1

        # Latest is always saved so an interrupted run can resume.
        save_training_checkpoint(
            checkpoint_path=FINE_TUNE_LATEST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            epoch_number=epoch_number,
            breed_names=breed_names,
            validation_loss=validation_loss,
            validation_accuracy=validation_accuracy,
            best_validation_loss=best_validation_loss,
        )

        print(
            "Saved latest fine-tuning checkpoint: "
            f"{FINE_TUNE_LATEST_CHECKPOINT_PATH}"
        )

        if epochs_without_accuracy_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation accuracy did not improve "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs"
            )
            break


if __name__ == "__main__":
    main()
