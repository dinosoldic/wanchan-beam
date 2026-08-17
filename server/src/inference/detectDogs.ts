import {
  processDetectorOutput,
  type DogDetection,
} from "../postprocessing/index.js";
import { preprocessDetectorImage } from "../preprocessing/index.js";

import { loadDetector } from "./loadDetector.js";

export interface DetectedImage {
  width: number;
  height: number;
}

export interface DogDetectionResult {
  image: DetectedImage;
  detections: DogDetection[];
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

  return {
    image: {
      width: transform.originalWidth,
      height: transform.originalHeight,
    },
    detections,
  };
}
