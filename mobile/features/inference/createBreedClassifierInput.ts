import type { LiveDetectionBox } from "./decodeMobileDetectorOutput";

const DETECTOR_INPUT_SIZE = 544;
const DETECTOR_CHANNELS = 3;

export const BREED_CLASSIFIER_INPUT_SIZE = 256;
export const BREED_CLASSIFIER_OUTPUT_CLASSES = 130;

export const BREED_CLASSIFIER_INPUT_BYTE_LENGTH =
  BREED_CLASSIFIER_INPUT_SIZE *
  BREED_CLASSIFIER_INPUT_SIZE *
  DETECTOR_CHANNELS *
  Float32Array.BYTES_PER_ELEMENT;

export const BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH =
  BREED_CLASSIFIER_OUTPUT_CLASSES * Float32Array.BYTES_PER_ELEMENT;

// Keep these values aligned with models/v1/preprocessing.json.
const CHANNEL_MEANS = [0.485, 0.456, 0.406] as const;
const CHANNEL_STANDARD_DEVIATIONS = [0.229, 0.224, 0.225] as const;
const PADDING_RGB = [124 / 255, 116 / 255, 104 / 255] as const;

function clamp(value: number, minimum: number, maximum: number): number {
  "worklet";

  return Math.min(Math.max(value, minimum), maximum);
}

function roundHalfToEven(value: number): number {
  "worklet";

  const lowerInteger = Math.floor(value);
  const fraction = value - lowerInteger;

  if (fraction < 0.5) {
    return lowerInteger;
  }

  if (fraction > 0.5) {
    return lowerInteger + 1;
  }

  return lowerInteger % 2 === 0 ? lowerInteger : lowerInteger + 1;
}

export function createBreedClassifierInput(
  detectorInputBuffer: ArrayBuffer,
  detectionBox: LiveDetectionBox,
): ArrayBuffer {
  "worklet";

  const expectedDetectorBytes =
    DETECTOR_INPUT_SIZE *
    DETECTOR_INPUT_SIZE *
    DETECTOR_CHANNELS *
    Float32Array.BYTES_PER_ELEMENT;

  if (detectorInputBuffer.byteLength !== expectedDetectorBytes) {
    throw new Error(
      `Expected ${expectedDetectorBytes} detector input bytes, ` +
        `received ${detectorInputBuffer.byteLength}.`,
    );
  }

  // Round outward to match training and server crops.
  const cropLeft = clamp(Math.floor(detectionBox.x1), 0, DETECTOR_INPUT_SIZE);
  const cropTop = clamp(Math.floor(detectionBox.y1), 0, DETECTOR_INPUT_SIZE);
  const cropRight = clamp(Math.ceil(detectionBox.x2), 0, DETECTOR_INPUT_SIZE);
  const cropBottom = clamp(Math.ceil(detectionBox.y2), 0, DETECTOR_INPUT_SIZE);

  const cropWidth = cropRight - cropLeft;
  const cropHeight = cropBottom - cropTop;

  if (cropWidth <= 0 || cropHeight <= 0) {
    throw new Error("Breed classifier crop has no usable area.");
  }

  const resizeScale = Math.min(
    BREED_CLASSIFIER_INPUT_SIZE / cropWidth,
    BREED_CLASSIFIER_INPUT_SIZE / cropHeight,
  );

  const resizedWidth = Math.max(1, roundHalfToEven(cropWidth * resizeScale));
  const resizedHeight = Math.max(1, roundHalfToEven(cropHeight * resizeScale));

  const paddingLeft = Math.floor(
    (BREED_CLASSIFIER_INPUT_SIZE - resizedWidth) / 2,
  );
  const paddingTop = Math.floor(
    (BREED_CLASSIFIER_INPUT_SIZE - resizedHeight) / 2,
  );

  const detectorValues = new Float32Array(detectorInputBuffer);
  const classifierBuffer = new ArrayBuffer(BREED_CLASSIFIER_INPUT_BYTE_LENGTH);
  const classifierValues = new Float32Array(classifierBuffer);

  const detectorPlaneSize = DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE;
  const classifierPlaneSize =
    BREED_CLASSIFIER_INPUT_SIZE * BREED_CLASSIFIER_INPUT_SIZE;

  for (let channel = 0; channel < DETECTOR_CHANNELS; channel += 1) {
    const mean = CHANNEL_MEANS[channel]!;
    const standardDeviation = CHANNEL_STANDARD_DEVIATIONS[channel]!;
    const normalizedPadding =
      (PADDING_RGB[channel]! - mean) / standardDeviation;

    const classifierPlaneOffset = channel * classifierPlaneSize;
    const detectorPlaneOffset = channel * detectorPlaneSize;

    classifierValues.fill(
      normalizedPadding,
      classifierPlaneOffset,
      classifierPlaneOffset + classifierPlaneSize,
    );

    for (let resizedY = 0; resizedY < resizedHeight; resizedY += 1) {
      // Map pixel centers for bilinear resizing.
      const sourceY = clamp(
        ((resizedY + 0.5) * cropHeight) / resizedHeight - 0.5,
        0,
        cropHeight - 1,
      );

      const sourceY0 = Math.floor(sourceY);
      const sourceY1 = Math.min(sourceY0 + 1, cropHeight - 1);
      const verticalWeight = sourceY - sourceY0;

      const absoluteY0 = cropTop + sourceY0;
      const absoluteY1 = cropTop + sourceY1;
      const outputY = paddingTop + resizedY;

      for (let resizedX = 0; resizedX < resizedWidth; resizedX += 1) {
        const sourceX = clamp(
          ((resizedX + 0.5) * cropWidth) / resizedWidth - 0.5,
          0,
          cropWidth - 1,
        );

        const sourceX0 = Math.floor(sourceX);
        const sourceX1 = Math.min(sourceX0 + 1, cropWidth - 1);
        const horizontalWeight = sourceX - sourceX0;

        const absoluteX0 = cropLeft + sourceX0;
        const absoluteX1 = cropLeft + sourceX1;

        const topLeft =
          detectorValues[
            detectorPlaneOffset + absoluteY0 * DETECTOR_INPUT_SIZE + absoluteX0
          ]!;

        const topRight =
          detectorValues[
            detectorPlaneOffset + absoluteY0 * DETECTOR_INPUT_SIZE + absoluteX1
          ]!;

        const bottomLeft =
          detectorValues[
            detectorPlaneOffset + absoluteY1 * DETECTOR_INPUT_SIZE + absoluteX0
          ]!;

        const bottomRight =
          detectorValues[
            detectorPlaneOffset + absoluteY1 * DETECTOR_INPUT_SIZE + absoluteX1
          ]!;

        const topValue = topLeft + (topRight - topLeft) * horizontalWeight;

        const bottomValue =
          bottomLeft + (bottomRight - bottomLeft) * horizontalWeight;

        const resizedValue =
          topValue + (bottomValue - topValue) * verticalWeight;

        const outputX = paddingLeft + resizedX;
        const outputIndex =
          classifierPlaneOffset +
          outputY * BREED_CLASSIFIER_INPUT_SIZE +
          outputX;

        classifierValues[outputIndex] =
          (resizedValue - mean) / standardDeviation;
      }
    }
  }

  return classifierBuffer;
}
