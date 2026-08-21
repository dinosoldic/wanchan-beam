import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";

const EXPECTED_BREED_COUNT = 130;
const EXPECTED_INPUT_SHAPE = ["batch", 3, 256, 256] as const;
const EXPECTED_OUTPUT_SHAPE = ["batch", 130] as const;

const defaultModelPath = fileURLToPath(
  new URL("../../../models/v1/breed-classifier.onnx", import.meta.url),
);

const defaultLabelsPath = fileURLToPath(
  new URL("../../../models/v1/labels.json", import.meta.url),
);

const modelPath = process.env.BREED_CLASSIFIER_MODEL_PATH ?? defaultModelPath;

const labelsPath =
  process.env.BREED_CLASSIFIER_LABELS_PATH ?? defaultLabelsPath;

export interface BreedClassifierAssets {
  readonly session: ort.InferenceSession;
  readonly labels: readonly string[];
}

let assetsPromise: Promise<BreedClassifierAssets> | undefined;

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function shapesMatch(
  actual: readonly (number | string)[],
  expected: readonly (number | string)[],
): boolean {
  return (
    actual.length === expected.length &&
    actual.every((dimension, index) => dimension === expected[index])
  );
}

function parseLabels(contents: string): readonly string[] {
  const parsedLabels: unknown = JSON.parse(contents);

  if (!Array.isArray(parsedLabels)) {
    throw new TypeError("Breed labels must be a JSON array");
  }

  if (!parsedLabels.every(isNonEmptyString)) {
    throw new TypeError("Every breed label must be a non-empty string");
  }

  if (parsedLabels.length !== EXPECTED_BREED_COUNT) {
    throw new Error(
      "Unexpected breed-label count: " +
        `expected ${EXPECTED_BREED_COUNT}, ` +
        `got ${parsedLabels.length}`,
    );
  }

  if (new Set(parsedLabels).size !== parsedLabels.length) {
    throw new Error("Breed labels must be unique");
  }

  return Object.freeze([...parsedLabels]);
}

function validateSession(session: ort.InferenceSession): void {
  if (
    session.inputMetadata.length !== 1 ||
    session.outputMetadata.length !== 1
  ) {
    throw new Error(
      "Breed classifier must have exactly one input and one output",
    );
  }

  const input = session.inputMetadata[0];
  const output = session.outputMetadata[0];

  if (!input?.isTensor) {
    throw new TypeError("Breed classifier input must be a tensor");
  }

  if (
    input.name !== "images" ||
    input.type !== "float32" ||
    !shapesMatch(input.shape, EXPECTED_INPUT_SHAPE)
  ) {
    throw new Error(
      `Unexpected classifier input metadata: ${JSON.stringify(input)}`,
    );
  }

  if (!output?.isTensor) {
    throw new TypeError("Breed classifier output must be a tensor");
  }

  if (
    output.name !== "logits" ||
    output.type !== "float32" ||
    !shapesMatch(output.shape, EXPECTED_OUTPUT_SHAPE)
  ) {
    throw new Error(
      `Unexpected classifier output metadata: ${JSON.stringify(output)}`,
    );
  }
}

async function createBreedClassifierAssets(): Promise<BreedClassifierAssets> {
  const labelContents = await readFile(labelsPath, "utf8");
  const labels = parseLabels(labelContents);

  const session = await ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });

  validateSession(session);

  return Object.freeze({
    session,
    labels,
  });
}

export function loadBreedClassifier(): Promise<BreedClassifierAssets> {
  assetsPromise ??= createBreedClassifierAssets();

  return assetsPromise;
}
