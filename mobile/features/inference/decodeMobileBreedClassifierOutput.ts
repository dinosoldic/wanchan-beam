import { BREED_CLASSIFIER_OUTPUT_CLASSES } from "./createBreedClassifierInput";

export interface LiveBreedPrediction {
  classId: number;
  confidence: number;
}

export function decodeMobileBreedClassifierOutput(
  outputBuffer: ArrayBuffer,
): LiveBreedPrediction {
  "worklet";

  const logits = new Float32Array(outputBuffer);

  if (logits.length !== BREED_CLASSIFIER_OUTPUT_CLASSES) {
    throw new Error(
      `Expected ${BREED_CLASSIFIER_OUTPUT_CLASSES} breed logits, ` +
        `received ${logits.length}.`,
    );
  }

  let topClassId = 0;
  let maximumLogit = logits[0]!;

  for (
    let classId = 1;
    classId < BREED_CLASSIFIER_OUTPUT_CLASSES;
    classId += 1
  ) {
    const logit = logits[classId]!;

    if (!Number.isFinite(logit)) {
      throw new Error(`Breed logit ${classId} is not finite.`);
    }

    if (logit > maximumLogit) {
      maximumLogit = logit;
      topClassId = classId;
    }
  }

  if (!Number.isFinite(maximumLogit)) {
    throw new Error("Breed logit 0 is not finite.");
  }

  // Subtracting the maximum keeps softmax numerically stable. Because the
  // winning logit's exponent is exactly one, its probability is 1 / the sum.
  let exponentialSum = 0;

  for (
    let classId = 0;
    classId < BREED_CLASSIFIER_OUTPUT_CLASSES;
    classId += 1
  ) {
    exponentialSum += Math.exp(logits[classId]! - maximumLogit);
  }

  if (!Number.isFinite(exponentialSum) || exponentialSum <= 0) {
    throw new Error("Breed softmax produced an invalid sum.");
  }

  return {
    classId: topClassId,
    confidence: 1 / exponentialSum,
  };
}
