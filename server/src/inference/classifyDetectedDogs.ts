import type {
  DogBreedPredictions,
  DogDetection,
} from "../postprocessing/index.js";
import { letterboxBreedCrop } from "../preprocessing/index.js";

import { classifyBreeds } from "./classifyBreeds.js";

export interface ClassifiedDogDetection extends DogDetection {
  breedPredictions: DogBreedPredictions;
}

export async function classifyDetectedDogs(
  imageBuffer: Buffer,
  detections: readonly DogDetection[],
): Promise<ClassifiedDogDetection[]> {
  // An empty crop batch is invalid and there is nothing to classify.
  if (detections.length === 0) {
    return [];
  }

  // Promise.all preserves detection order while preparing the crops concurrently.
  const crops = await Promise.all(
    detections.map(async ({ box }) => {
      const { pixels } = await letterboxBreedCrop(imageBuffer, box);

      return pixels;
    }),
  );

  // Run one classifier call for the complete crop batch.
  const predictions = await classifyBreeds(crops);

  return detections.map((detection, index) => {
    const breedPredictions = predictions[index];

    if (!breedPredictions) {
      throw new Error(`Missing breed predictions for detection ${index}`);
    }

    return {
      ...detection,
      breedPredictions,
    };
  });
}
