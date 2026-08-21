import assert from "node:assert/strict";
import { test } from "node:test";

import { createBreedClassifierTensor } from "../src/preprocessing/index.js";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const PIXEL_COUNT = INPUT_SIZE * INPUT_SIZE;
const EXPECTED_BYTE_COUNT = PIXEL_COUNT * CHANNEL_COUNT;

function assertApproximatelyEqual(
  actual: number,
  expected: number,
  tolerance = 0.000001,
): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `Expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

function createSolidCrop(red: number, green: number, blue: number): Uint8Array {
  const pixels = new Uint8Array(EXPECTED_BYTE_COUNT);

  for (let pixelIndex = 0; pixelIndex < PIXEL_COUNT; pixelIndex += 1) {
    const offset = pixelIndex * CHANNEL_COUNT;

    pixels[offset] = red;
    pixels[offset + 1] = green;
    pixels[offset + 2] = blue;
  }

  return pixels;
}

test("creates a normalized NCHW tensor", () => {
  const pixels = new Uint8Array(EXPECTED_BYTE_COUNT);

  // Give the first two pixels distinct channel values so the test can
  // distinguish interleaved RGB input from planar NCHW output.
  pixels.set([0, 127, 255, 255, 0, 127], 0);

  const tensor = createBreedClassifierTensor([pixels]);

  assert.equal(tensor.type, "float32");
  assert.deepEqual(tensor.dims, [1, CHANNEL_COUNT, INPUT_SIZE, INPUT_SIZE]);

  const tensorData = tensor.data as Float32Array;

  assert.equal(tensorData.length, EXPECTED_BYTE_COUNT);

  // First source pixel: RGB (0, 127, 255).
  assertApproximatelyEqual(tensorData[0]!, -2.11790394783);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT]!, 0.187675058842);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT * 2]!, 2.6400001049);

  // Second source pixel: RGB (255, 0, 127).
  assertApproximatelyEqual(tensorData[1]!, 2.24890828133);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT + 1]!, -2.03571414948);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT * 2 + 1]!, 0.409063249826);
});

test("keeps multiple crops in separate batch regions", () => {
  const firstCrop = createSolidCrop(10, 20, 30);
  const secondCrop = createSolidCrop(255, 127, 0);

  const tensor = createBreedClassifierTensor([firstCrop, secondCrop]);

  assert.deepEqual(tensor.dims, [2, CHANNEL_COUNT, INPUT_SIZE, INPUT_SIZE]);

  const tensorData = tensor.data as Float32Array;
  const secondBatchOffset = EXPECTED_BYTE_COUNT;

  // Verify the first pixel of batch item zero.
  assertApproximatelyEqual(tensorData[0]!, -1.94665646553);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT]!, -1.68557417393);
  assertApproximatelyEqual(tensorData[PIXEL_COUNT * 2]!, -1.28156864643);

  // Verify the first pixel of batch item one.
  assertApproximatelyEqual(tensorData[secondBatchOffset]!, 2.24890828133);
  assertApproximatelyEqual(
    tensorData[secondBatchOffset + PIXEL_COUNT]!,
    0.187675058842,
  );
  assertApproximatelyEqual(
    tensorData[secondBatchOffset + PIXEL_COUNT * 2]!,
    -1.80444443226,
  );
});

test("rejects an empty crop batch", () => {
  assert.throws(() => createBreedClassifierTensor([]), {
    name: "RangeError",
    message: "At least one breed crop is required",
  });
});

test("identifies an incorrectly sized crop by batch index", () => {
  const validCrop = createSolidCrop(10, 20, 30);

  const invalidCrop = new Uint8Array(3);

  assert.throws(() => createBreedClassifierTensor([validCrop, invalidCrop]), {
    message: "Breed crop 1 must contain " + "196608 RGB bytes, received 3",
  });
});
