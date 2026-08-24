import { processDetectorOutput } from "../postprocessing/index.js";
import { preprocessDetectorImage } from "../preprocessing/index.js";

import {
  classifyDetectedDogs,
  type ClassifiedDogDetection,
} from "./classifyDetectedDogs.js";
import { loadDetector } from "./loadDetector.js";

export interface DetectedImage {
  width: number;
  height: number;
}

export interface DogDetectionResult {
  image: DetectedImage;
  detections: ClassifiedDogDetection[];
}

export async function detectDogs(
  imageBuffer: Buffer,
): Promise<DogDetectionResult> {
  const { tensor, transform } = await preprocessDetectorImage(imageBuffer);

  const detector = await loadDetector();

  const outputs = await detector.run({
    images: tensor,
  });

  const output = outputs.output0;

  if (!output) {
    throw new Error("Detector did not return output0");
  }

  const detections = processDetectorOutput(output, transform);

  // Crop every dog and classify the crops together.
  const classifiedDetections = await classifyDetectedDogs(
    imageBuffer,
    detections,
  );

  return {
    image: {
      width: transform.originalWidth,
      height: transform.originalHeight,
    },
    detections: classifiedDetections,
  };
}
