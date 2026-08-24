"""Compare PyTorch and mobile LiteRT outputs on all validation crops."""

from time import perf_counter

# LiteRT is installed only inside the Linux export container.
from ai_edge_litert.interpreter import (  # pyright: ignore[reportMissingImports]
    Interpreter,
)
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.tsinghua_dogs import (
    TRAIN_SPLIT_PATH,
    VALIDATION_SPLIT_PATH,
    TsinghuaDogsDataset,
    build_breed_mapping,
    load_split_paths,
)
from evaluation.evaluate_breed_classifier import load_model_checkpoint
from export.export_mobile_breed_classifier import (
    INPUT_SIZE,
    NUMBER_OF_BREEDS,
    OUTPUT_MODEL_PATH,
    SOURCE_MODEL_PATH,
)

BATCH_SIZE = 64

# Avoid Docker shared-memory pressure from multiprocessing data loaders.
NUM_WORKERS = 0

# Allow small drift from different floating-point operation orders.
MAXIMUM_ALLOWED_ABSOLUTE_DIFFERENCE = 0.0002
MAXIMUM_ALLOWED_MEAN_DIFFERENCE = 0.00002


def main() -> None:
    """Compare the complete validation split across both runtimes."""

    training_paths = load_split_paths(TRAIN_SPLIT_PATH)
    breed_names, breed_to_id = build_breed_mapping(training_paths)

    model, _, classifier_hidden_features, model_input_size = load_model_checkpoint(
        checkpoint_path=SOURCE_MODEL_PATH,
        expected_breed_names=breed_names,
        device=torch.device("cpu"),
    )

    if model_input_size != INPUT_SIZE:
        raise ValueError(f"Expected input size {INPUT_SIZE}, got {model_input_size}")

    validation_paths = load_split_paths(VALIDATION_SPLIT_PATH)
    validation_dataset = TsinghuaDogsDataset(
        validation_paths,
        breed_to_id,
        model_input_size=model_input_size,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    if not OUTPUT_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Mobile classifier not found: {OUTPUT_MODEL_PATH}")

    interpreter = Interpreter(model_path=str(OUTPUT_MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    if len(input_details) != 1:
        raise ValueError(f"Expected one LiteRT input, found {len(input_details)}.")

    if len(output_details) != 1:
        raise ValueError(f"Expected one LiteRT output, found {len(output_details)}.")

    input_detail = input_details[0]
    output_detail = output_details[0]

    input_shape = tuple(int(value) for value in input_detail["shape"])
    output_shape = tuple(int(value) for value in output_detail["shape"])

    expected_input_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
    expected_output_shape = (1, NUMBER_OF_BREEDS)

    if input_shape != expected_input_shape:
        raise ValueError(
            f"Expected input shape {expected_input_shape}, got {input_shape}."
        )

    if output_shape != expected_output_shape:
        raise ValueError(
            f"Expected output shape {expected_output_shape}, got {output_shape}."
        )

    if np.dtype(input_detail["dtype"]) != np.dtype(np.float32):
        raise ValueError(f"Expected float32 input, got {input_detail['dtype']}.")

    if np.dtype(output_detail["dtype"]) != np.dtype(np.float32):
        raise ValueError(f"Expected float32 output, got {output_detail['dtype']}.")

    input_index = int(input_detail["index"])
    output_index = int(output_detail["index"])

    maximum_absolute_difference = 0.0
    total_absolute_difference = 0.0
    total_logit_values = 0
    processed_examples = 0
    top_one_matches = 0
    top_two_matches = 0

    started_at = perf_counter()

    with torch.inference_mode():
        for batch_index, (batch_images, _) in enumerate(
            validation_loader,
            start=1,
        ):
            pytorch_logits_tensor = model(batch_images)
            pytorch_logits = pytorch_logits_tensor.numpy()

            batch_size = int(batch_images.shape[0])
            expected_shape = (batch_size, NUMBER_OF_BREEDS)

            if pytorch_logits.shape != expected_shape:
                raise ValueError(
                    f"Unexpected PyTorch output shape: {pytorch_logits.shape}"
                )

            batch_images_numpy = batch_images.numpy()
            litert_logits = np.empty(expected_shape, dtype=np.float32)

            # Reuse the fixed batch-one interpreter for each image.
            for example_index in range(batch_size):
                example_input = np.ascontiguousarray(
                    batch_images_numpy[example_index : example_index + 1],
                    dtype=np.float32,
                )

                interpreter.set_tensor(input_index, example_input)
                interpreter.invoke()

                example_logits = interpreter.get_tensor(output_index)

                if not isinstance(example_logits, np.ndarray):
                    raise TypeError(
                        "Expected dense NumPy logits, but LiteRT returned "
                        f"{type(example_logits).__name__}."
                    )

                if example_logits.shape != expected_output_shape:
                    raise ValueError(
                        "Unexpected LiteRT output shape: " f"{example_logits.shape}."
                    )

                litert_logits[example_index] = example_logits[0]

            absolute_difference = np.abs(pytorch_logits - litert_logits)

            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(absolute_difference.max()),
            )
            total_absolute_difference += float(
                absolute_difference.sum(dtype=np.float64)
            )
            total_logit_values += int(absolute_difference.size)

            pytorch_top_two = torch.topk(
                pytorch_logits_tensor,
                k=2,
                dim=1,
            ).indices.numpy()

            litert_top_two = np.argsort(
                -litert_logits,
                axis=1,
            )[:, :2]

            top_one_matches += int(
                np.count_nonzero(pytorch_top_two[:, 0] == litert_top_two[:, 0])
            )
            top_two_matches += int(
                np.count_nonzero(
                    np.all(
                        pytorch_top_two == litert_top_two,
                        axis=1,
                    )
                )
            )

            processed_examples += batch_size

            if batch_index % 10 == 0 or processed_examples == len(validation_dataset):
                print(
                    f"Compared {processed_examples}/"
                    f"{len(validation_dataset)} validation examples"
                )

    if processed_examples != len(validation_dataset):
        raise RuntimeError("Parity verification did not process the complete dataset")

    mean_absolute_difference = total_absolute_difference / total_logit_values
    elapsed_seconds = perf_counter() - started_at

    print()
    print(f"Validation examples compared: {processed_examples}")
    print(f"Classifier hidden features: {classifier_hidden_features}")
    print(f"LiteRT output shape: [1, {NUMBER_OF_BREEDS}]")
    print(f"Maximum absolute logit difference: {maximum_absolute_difference:.10f}")
    print(f"Mean absolute logit difference: {mean_absolute_difference:.10f}")
    print(f"Matching top-1 IDs: {top_one_matches}/{processed_examples}")
    print(f"Matching ordered top-2 IDs: {top_two_matches}/{processed_examples}")
    print(f"Elapsed time: {elapsed_seconds:.1f} seconds")

    if maximum_absolute_difference > MAXIMUM_ALLOWED_ABSOLUTE_DIFFERENCE:
        raise AssertionError(
            "Maximum logit difference exceeded the allowed limit: "
            f"{maximum_absolute_difference:.10f}"
        )

    if mean_absolute_difference > MAXIMUM_ALLOWED_MEAN_DIFFERENCE:
        raise AssertionError(
            "Mean logit difference exceeded the allowed limit: "
            f"{mean_absolute_difference:.10f}"
        )

    if top_one_matches != processed_examples:
        raise AssertionError("PyTorch and LiteRT top-1 predictions did not all match")

    if top_two_matches != processed_examples:
        raise AssertionError(
            "PyTorch and LiteRT ordered top-2 predictions did not all match"
        )

    print("Full PyTorch/LiteRT mobile classifier parity validation passed.")


if __name__ == "__main__":
    main()
