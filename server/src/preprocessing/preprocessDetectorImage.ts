import type { Tensor } from "onnxruntime-node";

import { createDetectorTensor } from "./createDetectorTensor.js";
import { letterboxImage, type LetterboxTransform } from "./letterboxImage.js";

export interface PreprocessedDetectorImage {
  tensor: Tensor;
  transform: LetterboxTransform;
}

export async function preprocessDetectorImage(
  input: Buffer,
): Promise<PreprocessedDetectorImage> {
  const { pixels, transform } = await letterboxImage(input);

  return {
    tensor: createDetectorTensor(pixels),
    transform,
  };
}
