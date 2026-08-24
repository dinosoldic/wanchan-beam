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
  // Skip the classifier when no dogs were found.
  if (detections.length === 0) {
    return [];
  }

  // Prepare crops together without changing detection order.
  const crops = await Promise.all(
    detections.map(async ({ box }) => {
      const { pixels } = await letterboxBreedCrop(imageBuffer, box);

      return pixels;
    }),
  );

  // Run one classifier call for the complete batch.
  // console.log(`Classifying breeds for ${crops.length} detected dog(s)...`);

  const predictions = await classifyBreeds(crops);

  // Optional output for checking both breed predictions.
  // console.table(
  //   predictions.map(([firstBreed, secondBreed], index) => ({
  //     dog: index + 1,
  //     firstBreed: firstBreed.label,
  //     firstConfidence: `${(firstBreed.confidence * 100).toFixed(1)}%`,
  //     secondBreed: secondBreed.label,
  //     secondConfidence: `${(secondBreed.confidence * 100).toFixed(1)}%`,
  //   })),
  // );

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
