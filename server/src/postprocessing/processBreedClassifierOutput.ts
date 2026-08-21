import type { Tensor } from "onnxruntime-node";

const BREED_COUNT = 130;

export interface BreedPrediction {
  classId: number;
  label: string;
  confidence: number;
}

export type DogBreedPredictions = readonly [BreedPrediction, BreedPrediction];

function requireLabel(labels: readonly string[], classId: number): string {
  const label = labels[classId];

  if (label === undefined) {
    throw new Error(`Missing breed label for class ID ${classId}`);
  }

  return label;
}

export function processBreedClassifierOutput(
  output: Tensor,
  labels: readonly string[],
): DogBreedPredictions[] {
  if (labels.length !== BREED_COUNT) {
    throw new Error(
      "Unexpected breed-label count: " +
        `expected ${BREED_COUNT}, got ${labels.length}`,
    );
  }

  if (output.type !== "float32") {
    throw new Error(
      "Expected float32 classifier output, " + `received ${output.type}`,
    );
  }

  if (output.dims.length !== 2) {
    throw new Error(
      "Expected classifier output rank 2, " +
        `received shape [${output.dims.join(", ")}]`,
    );
  }

  const batchSize = output.dims[0]!;
  const breedCount = output.dims[1]!;

  if (
    !Number.isInteger(batchSize) ||
    batchSize <= 0 ||
    breedCount !== BREED_COUNT
  ) {
    throw new Error(
      "Unexpected classifier output shape: " + `[${output.dims.join(", ")}]`,
    );
  }

  const logits = output.data as Float32Array;
  const expectedValueCount = batchSize * BREED_COUNT;

  if (logits.length !== expectedValueCount) {
    throw new Error(
      `Expected ${expectedValueCount} classifier logits, ` +
        `received ${logits.length}`,
    );
  }

  const batchPredictions: DogBreedPredictions[] = [];

  for (let batchIndex = 0; batchIndex < batchSize; batchIndex += 1) {
    const batchOffset = batchIndex * BREED_COUNT;

    let maximumLogit = Number.NEGATIVE_INFINITY;

    let topOneId = -1;
    let topOneLogit = Number.NEGATIVE_INFINITY;

    let topTwoId = -1;
    let topTwoLogit = Number.NEGATIVE_INFINITY;

    for (let classId = 0; classId < BREED_COUNT; classId += 1) {
      const logit = logits[batchOffset + classId]!;

      if (!Number.isFinite(logit)) {
        throw new Error(
          "Classifier returned a non-finite logit " +
            `for batch ${batchIndex}, class ${classId}`,
        );
      }

      maximumLogit = Math.max(maximumLogit, logit);

      // Strict comparisons preserve lower class IDs when logits tie,
      // because classes are visited in ascending ID order.
      if (logit > topOneLogit) {
        topTwoId = topOneId;
        topTwoLogit = topOneLogit;

        topOneId = classId;
        topOneLogit = logit;
      } else if (logit > topTwoLogit) {
        topTwoId = classId;
        topTwoLogit = logit;
      }
    }

    if (topOneId < 0 || topTwoId < 0) {
      throw new Error(`Could not select two breeds for batch ${batchIndex}`);
    }

    // Subtracting the maximum keeps exponentials within a safe range.
    let softmaxDenominator = 0;

    for (let classId = 0; classId < BREED_COUNT; classId += 1) {
      const logit = logits[batchOffset + classId]!;

      softmaxDenominator += Math.exp(logit - maximumLogit);
    }

    const topOneConfidence =
      Math.exp(topOneLogit - maximumLogit) / softmaxDenominator;

    const topTwoConfidence =
      Math.exp(topTwoLogit - maximumLogit) / softmaxDenominator;

    batchPredictions.push([
      {
        classId: topOneId,
        label: requireLabel(labels, topOneId),
        confidence: topOneConfidence,
      },
      {
        classId: topTwoId,
        label: requireLabel(labels, topTwoId),
        confidence: topTwoConfidence,
      },
    ]);
  }

  return batchPredictions;
}
