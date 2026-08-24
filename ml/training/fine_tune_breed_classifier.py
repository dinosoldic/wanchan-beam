"""Partially fine-tune the trained MobileNetV3 breed classifier."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_training_augmentation,
    build_training_tensor_augmentation,
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

BASE_EXPERIMENT_NAME = (
    "1module-clr-5e-5-flr-5e-6-cosine-ls-0.02-affine-" "wd-1e-2-dropout-0.4-input-256"
)

BASELINE_CHECKPOINT_PATH = CHECKPOINT_DIRECTORY / "head-baseline-best.pt"

NUMBER_OF_FEATURE_MODULES_TO_UNFREEZE = 1
CLASSIFIER_HIDDEN_FEATURES: int | None = 1280

BATCH_SIZE = 64
NUMBER_OF_WORKERS = 4
DEFAULT_RANDOM_SEED = 43
UNSUFFIXED_CHECKPOINT_SEED = 42
TOTAL_EPOCHS = 5
BATCH_LOG_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 2
RESUME_FROM_CHECKPOINT = True

# Let the new classifier adapt faster than the pretrained features.
CLASSIFIER_LEARNING_RATE = 0.00005
FEATURE_LEARNING_RATE = 0.000005
WEIGHT_DECAY = 0.01
LABEL_SMOOTHING = 0.02
USE_GEOMETRIC_AUGMENTATION = True
DROPOUT_PROBABILITY = 0.4
MODEL_INPUT_SIZE = 256
TRAIN_UNFROZEN_BATCH_NORM = False
MIXUP_ALPHA: float | None = None
CUTMIX_ALPHA: float | None = None
DEFAULT_RANDOM_ERASING_PROBABILITY = 0.25


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


def parse_arguments() -> tuple[int, float]:
    """Read optional overrides for the selected training configuration."""

    parser = argparse.ArgumentParser(
        description="Fine-tune the MobileNetV3 dog-breed classifier.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed used by training and balanced sampling.",
    )
    parser.add_argument(
        "--random-erasing-probability",
        type=float,
        default=DEFAULT_RANDOM_ERASING_PROBABILITY,
        help="Probability that one training image receives Random Erasing.",
    )

    arguments = parser.parse_args()

    if arguments.seed < 0:
        parser.error("--seed must be non-negative")

    if not 0.0 <= arguments.random_erasing_probability <= 1.0:
        parser.error("--random-erasing-probability must be between 0 and 1")

    return arguments.seed, arguments.random_erasing_probability


def build_experiment_name(
    random_seed: int,
    random_erasing_probability: float,
) -> str:
    """Build a unique run name while preserving existing seed-42 checkpoints."""

    experiment_name = BASE_EXPERIMENT_NAME

    if random_erasing_probability > 0.0:
        experiment_name += f"-random-erasing-p{random_erasing_probability:.2f}"

    # Seed 42 predates seed confirmation, so its checkpoints have no seed suffix.
    if random_seed != UNSUFFIXED_CHECKPOINT_SEED:
        experiment_name += f"-seed{random_seed}"

    return experiment_name


def main() -> None:
    """Fine-tune selected MobileNetV3 modules and its classifier."""

    random_seed, random_erasing_probability = parse_arguments()
    experiment_name = build_experiment_name(
        random_seed=random_seed,
        random_erasing_probability=random_erasing_probability,
    )

    fine_tune_latest_checkpoint_path = (
        CHECKPOINT_DIRECTORY / f"fine-tune-{experiment_name}-latest.pt"
    )
    fine_tune_best_loss_checkpoint_path = (
        CHECKPOINT_DIRECTORY / f"fine-tune-{experiment_name}-best-loss.pt"
    )
    fine_tune_best_accuracy_checkpoint_path = (
        CHECKPOINT_DIRECTORY / f"fine-tune-{experiment_name}-best-accuracy.pt"
    )

    set_random_seed(random_seed)

    device = select_device()

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    training_augmentation = build_training_augmentation(
        use_geometric_augmentation=USE_GEOMETRIC_AUGMENTATION,
    )

    training_tensor_augmentation = build_training_tensor_augmentation(
        random_erasing_probability=random_erasing_probability,
    )

    training_dataset = TsinghuaDogsDataset(
        training_paths,
        breed_to_id,
        image_augmentation=training_augmentation,
        tensor_augmentation=training_tensor_augmentation,
        model_input_size=MODEL_INPUT_SIZE,
    )

    # Keep balanced sampling reproducible.
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(random_seed)

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

    # Reject label ordering that would train the wrong output neurons.
    if tuple(baseline_checkpoint["breed_names"]) != breed_names:
        raise ValueError("Checkpoint breed ordering does not match the current dataset")

    model = build_breed_classifier(
        number_of_breeds=len(breed_names),
        use_pretrained_weights=False,
        dropout_probability=DROPOUT_PROBABILITY,
        classifier_hidden_features=CLASSIFIER_HIDDEN_FEATURES,
    ).to(device)

    model.load_state_dict(baseline_checkpoint["model_state_dict"])

    # Reopen only the deepest task-specific feature modules.
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

    # Use separate learning rates for new and pretrained parameters.
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

    # Reduce both learning rates after every epoch.
    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=TOTAL_EPOCHS,
        eta_min=0.0,
    )

    training_loss_function = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING,
    )

    validation_loss_function = nn.CrossEntropyLoss()

    # Use mixed precision to reduce CUDA memory use.
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

    if RESUME_FROM_CHECKPOINT and fine_tune_latest_checkpoint_path.exists():
        (
            completed_epoch,
            best_validation_loss,
            epochs_without_metric_improvement,
        ) = load_training_checkpoint(
            checkpoint_path=fine_tune_latest_checkpoint_path,
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

        if fine_tune_best_accuracy_checkpoint_path.exists():
            best_accuracy_checkpoint = torch.load(
                fine_tune_best_accuracy_checkpoint_path,
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

    # Catch trainable parameters missing from the optimizer.
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
    print(f"Experiment: {experiment_name}")
    print(f"Random seed: {random_seed}")
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
    print(f"MixUp alpha: {MIXUP_ALPHA}")
    print(f"CutMix alpha: {CUTMIX_ALPHA}")
    print(f"Random Erasing probability: {random_erasing_probability}")

    for epoch_number in range(starting_epoch, TOTAL_EPOCHS + 1):
        print()
        print(f"Epoch {epoch_number}/{TOTAL_EPOCHS}")
        learning_rate_description = f"classifier={optimizer.param_groups[0]['lr']:.2e}"
        if feature_parameter_group:
            learning_rate_description += (
                f", features={optimizer.param_groups[1]['lr']:.2e}"
            )
        print(f"Learning rates: {learning_rate_description}")

        # train on the balanced set
        average_training_loss = train_one_epoch(
            model=model,
            data_loader=training_data_loader,
            loss_function=training_loss_function,
            optimizer=optimizer,
            gradient_scaler=gradient_scaler,
            device=device,
            log_interval=BATCH_LOG_INTERVAL,
            train_unfrozen_batch_norm=TRAIN_UNFROZEN_BATCH_NORM,
            mixup_alpha=MIXUP_ALPHA,
            cutmix_alpha=CUTMIX_ALPHA,
        )

        # validate without updating parameters
        validation_loss, validation_accuracy = evaluate(
            model=model,
            data_loader=validation_data_loader,
            loss_function=validation_loss_function,
            device=device,
        )

        print(f"Training loss: {average_training_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_accuracy:.2%}")

        # advance the cosine schedule
        learning_rate_scheduler.step()

        validation_improved = validation_loss < best_validation_loss
        accuracy_improved = validation_accuracy > best_validation_accuracy

        # Keep training while loss or accuracy improves.
        if validation_improved or accuracy_improved:
            epochs_without_metric_improvement = 0
        else:
            epochs_without_metric_improvement += 1

        if validation_improved:
            best_validation_loss = validation_loss

            save_training_checkpoint(
                checkpoint_path=fine_tune_best_loss_checkpoint_path,
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
                f"{fine_tune_best_loss_checkpoint_path}"
            )

        # Keep separate best-loss and best-accuracy checkpoints.
        if accuracy_improved:
            best_validation_accuracy = validation_accuracy

            save_training_checkpoint(
                checkpoint_path=fine_tune_best_accuracy_checkpoint_path,
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
                f"{fine_tune_best_accuracy_checkpoint_path}"
            )
        # Always save latest for resume.
        save_training_checkpoint(
            checkpoint_path=fine_tune_latest_checkpoint_path,
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
            f"{fine_tune_latest_checkpoint_path}"
        )

        if epochs_without_metric_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: neither validation loss nor accuracy improved "
                f"for {EARLY_STOPPING_PATIENCE} consecutive epochs"
            )
            break


if __name__ == "__main__":
    main()
