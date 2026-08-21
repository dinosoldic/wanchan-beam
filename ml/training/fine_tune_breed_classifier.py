"""Partially fine-tune the trained MobileNetV3 breed classifier."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_training_augmentation,
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

EXPERIMENT_NAME = (
    "1module-clr-5e-5-flr-5e-6-cosine-ls-0.02-affine-" "wd-1e-2-dropout-0.4-input-256"
)

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

NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE = 1
CLASSIFIER_HIDDEN_FEATURES: int | None = 1280

BATCH_SIZE = 64
NUMBER_OF_WORKERS = 4
RANDOM_SEED = 42
TOTAL_EPOCHS = 5
BATCH_LOG_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 2
RESUME_FROM_CHECKPOINT = True

# The classifier can adapt faster, while the pretrained feature extractor uses a
# smaller learning rate to avoid destroying useful ImageNet features.
CLASSIFIER_LEARNING_RATE = 0.00005
FEATURE_LEARNING_RATE = 0.000005
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.02
USE_GEOMETRIC_AUGMENTATION = True
DROPOUT_PROBABILITY = 0.4
MODEL_INPUT_SIZE = 256
TRAIN_UNFROZEN_BATCH_NORM = False


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
    """Fine-tune selected MobileNetV3 modules and its classifier."""

    set_random_seed(RANDOM_SEED)

    device = select_device()

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    training_augmentation = build_training_augmentation(
        use_geometric_augmentation=USE_GEOMETRIC_AUGMENTATION,
    )

    training_dataset = TsinghuaDogsDataset(
        training_paths,
        breed_to_id,
        image_augmentation=training_augmentation,
        model_input_size=MODEL_INPUT_SIZE,
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
        model_input_size=MODEL_INPUT_SIZE,
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
        dropout_probability=DROPOUT_PROBABILITY,
        classifier_hidden_features=CLASSIFIER_HIDDEN_FEATURES,
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

    if NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE == 0:
        unfrozen_feature_modules = []
    else:
        unfrozen_feature_modules = feature_modules[
            -NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE:
        ]

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

    if feature_parameter_group:
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
    else:
        optimizer = torch.optim.AdamW(
            classifier_parameter_group,
            lr=CLASSIFIER_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    # Reduce both learning rates smoothly after each epoch so later updates
    # refine the model without moving far from its best pretrained features.
    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS,
        eta_min=0.0,
    )

    training_loss_function = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    validation_loss_function = nn.CrossEntropyLoss()

    # Mixed precision reduces GPU memory use and generally speeds up CUDA training.
    mixed_precision_enabled = device.type == "cuda"

    gradient_scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=mixed_precision_enabled,
    )

    starting_epoch = 1
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_metric_improvement = 0
    checkpoint_status = "No fine-tuning checkpoint found; starting from baseline"

    if RESUME_FROM_CHECKPOINT and FINE_TUNE_LATEST_CHECKPOINT_PATH.exists():
        (
            completed_epoch,
            best_validation_loss,
            epochs_without_metric_improvement,
        ) = load_training_checkpoint(
            checkpoint_path=FINE_TUNE_LATEST_CHECKPOINT_PATH,
            model=model,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            expected_breed_names=breed_names,
            device=device,
            expected_model_input_size=MODEL_INPUT_SIZE,
            learning_rate_scheduler=learning_rate_scheduler,
            sampler_generator=sampler_generator,
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

        checkpoint_status = f"Resuming fine-tuning after epoch {completed_epoch}"

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

    optimizer_description = f"classifier lr={CLASSIFIER_LEARNING_RATE}"
    if feature_parameter_group:
        optimizer_description += f", feature lr={FEATURE_LEARNING_RATE}"
    optimizer_description += f", weight decay={WEIGHT_DECAY}"

    dropout_description = (
        "inactive (linear head)"
        if CLASSIFIER_HIDDEN_FEATURES is None
        else str(DROPOUT_PROBABILITY)
    )

    ## debug info
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Fine-tuning device: {device_description}")
    print(f"Checkpoint: {checkpoint_status}")
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
        f"Data: {len(training_dataset):,} training samples "
        f"({len(training_data_loader):,} batches), "
        f"{len(validation_dataset):,} validation samples "
        f"({len(validation_data_loader):,} batches)"
    )
    print(f"Geometric augmentation: {USE_GEOMETRIC_AUGMENTATION}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Optimizer: {optimizer_description}")
    print(f"Mixed precision: {mixed_precision_enabled}")
    print(f"Dropout probability: {dropout_description}")
    print(f"Classifier hidden features: {CLASSIFIER_HIDDEN_FEATURES}")
    print(f"Model input size: {MODEL_INPUT_SIZE} x {MODEL_INPUT_SIZE}")
    print(f"Unfrozen BatchNorm adaptation: {TRAIN_UNFROZEN_BATCH_NORM}")

    for epoch_number in range(starting_epoch, TOTAL_EPOCHS + 1):
        print()
        print(f"Epoch {epoch_number}/{TOTAL_EPOCHS}")
        learning_rate_description = f"classifier={optimizer.param_groups[0]['lr']:.2e}"
        if feature_parameter_group:
            learning_rate_description += (
                f", features={optimizer.param_groups[1]['lr']:.2e}"
            )
        print(f"Learning rates: {learning_rate_description}")

        # Train on the complete balanced training set.
        average_training_loss = train_one_epoch(
            model=model,
            data_loader=training_data_loader,
            loss_function=training_loss_function,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            device=device,
            log_interval=BATCH_LOG_INTERVAL,
            train_unfrozen_batch_norm=TRAIN_UNFROZEN_BATCH_NORM,
        )

        # Measure generalization without updating any parameters.
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_data_loader,
            loss_function=validation_loss_function,
            device=device,
        )

        print(f"Training loss: {average_training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.2%}")

        # Advance the cosine schedule after completing the current epoch.
        learning_rate_scheduler.step()

        validation_improved = validation_loss < best_validation_loss
        accuracy_improved = validation_accuracy > best_validation_accuracy

        # Continue while either metric finds a better trade-off. Loss measures
        # confidence quality, while accuracy measures the final class decision.
        if validation_improved or accuracy_improved:
            epochs_without_metric_improvement = 0
        else:
            epochs_without_metric_improvement += 1

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
                learning_rate_scheduler=learning_rate_scheduler,
                sampler_generator=sampler_generator,
                early_stopping_counter=epochs_without_metric_improvement,
                model_input_size=MODEL_INPUT_SIZE,
            )

            print(
                "Saved new best-loss checkpoint: "
                f"{FINE_TUNE_BEST_LOSS_CHECKPOINT_PATH}"
            )

        # Accuracy and loss can favor different epochs, so preserve both.
        if accuracy_improved:
            best_validation_accuracy = validation_accuracy

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
                learning_rate_scheduler=learning_rate_scheduler,
                sampler_generator=sampler_generator,
                early_stopping_counter=epochs_without_metric_improvement,
                model_input_size=MODEL_INPUT_SIZE,
            )

            print(
                "Saved new best-accuracy checkpoint: "
                f"{FINE_TUNE_BEST_ACCURACY_CHECKPOINT_PATH}"
            )
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
            learning_rate_scheduler=learning_rate_scheduler,
            sampler_generator=sampler_generator,
            early_stopping_counter=epochs_without_metric_improvement,
            model_input_size=MODEL_INPUT_SIZE,
        )

        print(
            "Saved latest fine-tuning checkpoint: "
            f"{FINE_TUNE_LATEST_CHECKPOINT_PATH}"
        )

        if epochs_without_metric_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: neither validation loss nor accuracy improved "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs"
            )
            break


if __name__ == "__main__":
    main()
