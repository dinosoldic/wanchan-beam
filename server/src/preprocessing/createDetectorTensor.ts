import * as ort from "onnxruntime-node";

const INPUT_SIZE = 960;
const CHANNEL_COUNT = 3;
const PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
const EXPECTED_BYTE_COUNT = PIXEL_COUNT * CHANNEL_COUNT;

export function createDetectorTensor(pixels: Uint8Array): ort.Tensor {
  if (pixels.length !== EXPECTED_BYTE_COUNT) {
    throw new Error(
      `Expected ${EXPECTED_BYTE_COUNT} RGB bytes, received ${pixels.length}`,
    );
  }

  const tensorData = new Float32Array(EXPECTED_BYTE_COUNT);

  for (let pixelIndex = 0; pixelIndex < PIXEL_COUNT; pixelIndex += 1) {
    const sourceOffset = pixelIndex * CHANNEL_COUNT;

    const red = pixels[sourceOffset]!;
    const green = pixels[sourceOffset + 1]!;
    const blue = pixels[sourceOffset + 2]!;

    tensorData[pixelIndex] = red / 255;
    tensorData[PIXEL_COUNT + pixelIndex] = green / 255;
    tensorData[PIXEL_COUNT * 2 + pixelIndex] = blue / 255;
  }

  return new ort.Tensor("float32", tensorData, [
    1,
    CHANNEL_COUNT,
    INPUT_SIZE,
    INPUT_SIZE,
  ]);
}
