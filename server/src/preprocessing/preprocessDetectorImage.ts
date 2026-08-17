import type { Tensor } from "onnxruntime-node";

import { createDetectorTensor } from "./createDetectorTensor.js";
import { InvalidImageError } from "./InvalidImageError.js";
import {
  letterboxImage,
  type LetterboxedImage,
  type LetterboxTransform,
} from "./letterboxImage.js";

export interface PreprocessedDetectorImage {
  tensor: Tensor;
  transform: LetterboxTransform;
}

async function decodeAndLetterbox(input: Buffer): Promise<LetterboxedImage> {
  try {
    return await letterboxImage(input);
  } catch (cause) {
    throw new InvalidImageError(cause);
  }
}

export async function preprocessDetectorImage(
  input: Buffer,
): Promise<PreprocessedDetectorImage> {
  const { pixels, transform } = await decodeAndLetterbox(input);

  return {
    tensor: createDetectorTensor(pixels),
    transform,
  };
}
