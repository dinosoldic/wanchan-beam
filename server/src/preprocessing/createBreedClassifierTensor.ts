import * as ort from "onnxruntime-node";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
const EXPECTED_BYTE_COUNT = PIXEL_COUNT * CHANNEL_COUNT;

const RED_MEAN = Math.fround(0.485);
const GREEN_MEAN = Math.fround(0.456);
const BLUE_MEAN = Math.fround(0.406);

const RED_STANDARD_DEVIATION = Math.fround(0.229);
const GREEN_STANDARD_DEVIATION = Math.fround(0.224);
const BLUE_STANDARD_DEVIATION = Math.fround(0.225);

function normalizeChannel(
  byteValue: number,
  mean: number,
  standardDeviation: number,
): number {
  // Match ToTensor: convert an RGB byte from [0, 255] into [0, 1].
  const scaledValue = Math.fround(byteValue / 255);

  // Match torchvision Normalize using float32 operations.
  const centeredValue = Math.fround(scaledValue - mean);

  return Math.fround(centeredValue / standardDeviation);
}

export function createBreedClassifierTensor(
  crops: readonly Uint8Array[],
): ort.Tensor {
  if (crops.length === 0) {
    throw new RangeError("At least one breed crop is required");
  }

  const tensorData = new Float32Array(crops.length * EXPECTED_BYTE_COUNT);

  for (let batchIndex = 0; batchIndex < crops.length; batchIndex += 1) {
    const crop = crops[batchIndex]!;

    if (crop.length !== EXPECTED_BYTE_COUNT) {
      throw new Error(
        `Breed crop ${batchIndex} must contain ` +
          `${EXPECTED_BYTE_COUNT} RGB bytes, ` +
          `received ${crop.length}`,
      );
    }

    const batchOffset = batchIndex * EXPECTED_BYTE_COUNT;

    for (let pixelIndex = 0; pixelIndex < PIXEL_COUNT; pixelIndex += 1) {
      const sourceOffset = pixelIndex * CHANNEL_COUNT;

      const red = crop[sourceOffset]!;
      const green = crop[sourceOffset + 1]!;
      const blue = crop[sourceOffset + 2]!;

      tensorData[batchOffset + pixelIndex] = normalizeChannel(
        red,
        RED_MEAN,
        RED_STANDARD_DEVIATION,
      );

      tensorData[batchOffset + PIXEL_COUNT + pixelIndex] = normalizeChannel(
        green,
        GREEN_MEAN,
        GREEN_STANDARD_DEVIATION,
      );

      tensorData[batchOffset + PIXEL_COUNT * 2 + pixelIndex] = normalizeChannel(
        blue,
        BLUE_MEAN,
        BLUE_STANDARD_DEVIATION,
      );
    }
  }

  return new ort.Tensor("float32", tensorData, [
    crops.length,
    CHANNEL_COUNT,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
}
