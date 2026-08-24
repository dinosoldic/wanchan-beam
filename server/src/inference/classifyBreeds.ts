import {
  processBreedClassifierOutput,
  type DogBreedPredictions,
} from "../postprocessing/index.js";
import { createBreedClassifierTensor } from "../preprocessing/index.js";

import { loadBreedClassifier } from "./loadBreedClassifier.js";

export async function classifyBreeds(
  crops: readonly Uint8Array[],
): Promise<DogBreedPredictions[]> {
  // Classify all dog crops in one ONNX batch.
  const tensor = createBreedClassifierTensor(crops);

  const { session, labels } = await loadBreedClassifier();

  const outputs = await session.run({
    images: tensor,
  });

  const logits = outputs.logits;

  if (!logits) {
    throw new Error("Breed classifier did not return logits");
  }

  const predictions = processBreedClassifierOutput(logits, labels);

  if (predictions.length !== crops.length) {
    throw new Error(
      "Breed prediction count does not match crop count: " +
        `expected ${crops.length}, ` +
        `got ${predictions.length}`,
    );
  }

  return predictions;
}
