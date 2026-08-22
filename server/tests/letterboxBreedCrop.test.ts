import assert from "node:assert/strict";
import { test } from "node:test";

import sharp from "sharp";

import { letterboxBreedCrop } from "../src/preprocessing/index.js";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const EXPECTED_BYTE_COUNT = INPUT_SIZE * INPUT_SIZE * CHANNEL_COUNT;

interface RgbColor {
  r: number;
  g: number;
  b: number;
}

async function createSolidImage(
  width: number,
  height: number,
  background: RgbColor,
): Promise<Buffer> {
  return sharp({
    create: {
      width,
      height,
      channels: 3,
      background,
    },
  })
    .png()
    .toBuffer();
}

function readPixel(
  pixels: Uint8Array,
  x: number,
  y: number,
): [number, number, number] {
  const offset = (y * INPUT_SIZE + x) * CHANNEL_COUNT;

  return [pixels[offset]!, pixels[offset + 1]!, pixels[offset + 2]!];
}

test("crops and vertically pads a landscape dog box", async () => {
  const dogColor = {
    r: 10,
    g: 20,
    b: 30,
  };

  const image = await createSolidImage(100, 80, dogColor);

  const result = await letterboxBreedCrop(image, {
    x1: 9.2,
    y1: 19.7,
    x2: 90.1,
    y2: 60.2,
  });

  assert.equal(result.pixels.length, EXPECTED_BYTE_COUNT);

  assert.deepEqual(result.transform, {
    originalWidth: 100,
    originalHeight: 80,
    cropLeft: 9,
    cropTop: 19,
    cropWidth: 82,
    cropHeight: 42,
    resizedWidth: 256,
    resizedHeight: 131,
    paddingLeft: 0,
    paddingTop: 62,
    paddingRight: 0,
    paddingBottom: 63,
  });

  // The corner is padding, while the center belongs to the dog crop.
  assert.deepEqual(readPixel(result.pixels, 0, 0), [124, 116, 104]);
  assert.deepEqual(readPixel(result.pixels, 128, 128), [
    dogColor.r,
    dogColor.g,
    dogColor.b,
  ]);
});

test("clamps a fractional portrait box to the image bounds", async () => {
  const dogColor = {
    r: 20,
    g: 40,
    b: 60,
  };

  const image = await createSolidImage(50, 100, dogColor);

  const result = await letterboxBreedCrop(image, {
    x1: -5.4,
    y1: 9.2,
    x2: 40.1,
    y2: 110.9,
  });

  assert.deepEqual(result.transform, {
    originalWidth: 50,
    originalHeight: 100,
    cropLeft: 0,
    cropTop: 9,
    cropWidth: 41,
    cropHeight: 91,
    resizedWidth: 115,
    resizedHeight: 256,
    paddingLeft: 70,
    paddingTop: 0,
    paddingRight: 71,
    paddingBottom: 0,
  });

  assert.deepEqual(readPixel(result.pixels, 0, 128), [124, 116, 104]);
  assert.deepEqual(readPixel(result.pixels, 128, 128), [
    dogColor.r,
    dogColor.g,
    dogColor.b,
  ]);
});

test("matches Python half-to-even dimension rounding", async () => {
  const image = await createSolidImage(5, 512, {
    r: 80,
    g: 90,
    b: 100,
  });

  const result = await letterboxBreedCrop(image, {
    x1: 0,
    y1: 0,
    x2: 5,
    y2: 512,
  });

  // 5 × (256 / 512) equals exactly 2.5.
  // Python rounds 2.5 to the nearest even integer, which is 2.
  assert.equal(result.transform.resizedWidth, 2);
  assert.equal(result.transform.resizedHeight, 256);
  assert.equal(result.transform.paddingLeft, 127);
  assert.equal(result.transform.paddingRight, 127);
});

test("rejects a box with no original area", async () => {
  const image = await createSolidImage(100, 80, {
    r: 10,
    g: 20,
    b: 30,
  });

  await assert.rejects(
    () =>
      letterboxBreedCrop(image, {
        x1: 20,
        y1: 30,
        x2: 20,
        y2: 60,
      }),
    {
      name: "RangeError",
      message: "Breed crop box must have positive width and height",
    },
  );
});

test("rejects a box with no area after clamping", async () => {
  const image = await createSolidImage(100, 80, {
    r: 10,
    g: 20,
    b: 30,
  });

  await assert.rejects(
    () =>
      letterboxBreedCrop(image, {
        x1: 101,
        y1: 10,
        x2: 110,
        y2: 20,
      }),
    {
      name: "RangeError",
      message: "Breed crop box has no area after clamping",
    },
  );
});
