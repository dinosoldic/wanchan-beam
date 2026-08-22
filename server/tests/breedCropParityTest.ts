import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import { letterboxBreedCrop } from "../src/preprocessing/index.js";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const MAXIMUM_ALLOWED_CONTENT_DIFFERENCE = 2;
const MAXIMUM_ALLOWED_CONTENT_MEAN_DIFFERENCE = 0.2;

const sourceImagePath = fileURLToPath(
  new URL("../../ml/data/samples/test-dogs.png", import.meta.url),
);

const referenceImagePath = fileURLToPath(
  new URL("./fixtures/breed-crop-python-reference.png", import.meta.url),
);

const detectionBox = {
  x1: 565.9,
  y1: 131.5,
  x2: 774.5,
  y2: 422.5,
};

const [sourceImage, referenceImage] = await Promise.all([
  readFile(sourceImagePath),
  readFile(referenceImagePath),
]);

const [sharpResult, referenceResult] = await Promise.all([
  letterboxBreedCrop(sourceImage, detectionBox),
  sharp(referenceImage).toColourspace("srgb").raw().toBuffer({
    resolveWithObject: true,
  }),
]);

if (
  referenceResult.info.width !== INPUT_SIZE ||
  referenceResult.info.height !== INPUT_SIZE ||
  referenceResult.info.channels !== CHANNEL_COUNT
) {
  throw new Error(
    "Unexpected Python reference dimensions: " +
      `${referenceResult.info.width}x` +
      `${referenceResult.info.height}x` +
      `${referenceResult.info.channels}`,
  );
}

if (sharpResult.pixels.length !== referenceResult.data.length) {
  throw new Error("Sharp and Pillow output byte counts do not match");
}

let maximumDifference = 0;
let totalDifference = 0;
let identicalChannels = 0;

let contentMaximumDifference = 0;
let contentTotalDifference = 0;
let contentChannelCount = 0;
let identicalContentChannels = 0;
let contentDifferencesAboveOne = 0;
let contentDifferencesAboveTwo = 0;
let contentDifferencesAboveFive = 0;

let differentPaddingChannels = 0;

for (
  let channelIndex = 0;
  channelIndex < sharpResult.pixels.length;
  channelIndex += 1
) {
  const sharpValue = sharpResult.pixels[channelIndex]!;
  const referenceValue = referenceResult.data[channelIndex]!;
  const difference = Math.abs(sharpValue - referenceValue);

  maximumDifference = Math.max(maximumDifference, difference);
  totalDifference += difference;

  if (difference === 0) {
    identicalChannels += 1;
  }

  const pixelIndex = Math.floor(channelIndex / CHANNEL_COUNT);
  const x = pixelIndex % INPUT_SIZE;
  const y = Math.floor(pixelIndex / INPUT_SIZE);

  const insideContent =
    x >= sharpResult.transform.paddingLeft &&
    x < INPUT_SIZE - sharpResult.transform.paddingRight &&
    y >= sharpResult.transform.paddingTop &&
    y < INPUT_SIZE - sharpResult.transform.paddingBottom;

  if (insideContent) {
    contentMaximumDifference = Math.max(contentMaximumDifference, difference);
    contentTotalDifference += difference;
    contentChannelCount += 1;

    if (difference === 0) {
      identicalContentChannels += 1;
    }

    if (difference > 1) {
      contentDifferencesAboveOne += 1;
    }

    if (difference > 2) {
      contentDifferencesAboveTwo += 1;
    }

    if (difference > 5) {
      contentDifferencesAboveFive += 1;
    }
  } else if (difference !== 0) {
    differentPaddingChannels += 1;
  }
}

if (contentChannelCount === 0) {
  throw new Error("Breed crop parity comparison found no content pixels");
}

if (differentPaddingChannels !== 0) {
  throw new Error(
    "Sharp and Pillow padding pixels do not match: " +
      `${differentPaddingChannels} differing channels`,
  );
}

const meanDifference = totalDifference / sharpResult.pixels.length;

const contentMeanDifference = contentTotalDifference / contentChannelCount;

const identicalPercentage =
  (identicalChannels / sharpResult.pixels.length) * 100;

const identicalContentPercentage =
  (identicalContentChannels / contentChannelCount) * 100;

console.log("Sharp transform:", sharpResult.transform);
console.log(
  "Compared RGB channels:",
  sharpResult.pixels.length.toLocaleString(),
);
console.log(`Maximum byte difference: ${maximumDifference}`);
console.log(`Mean byte difference: ${meanDifference.toFixed(6)}`);
console.log(
  "Identical RGB channels: " +
    `${identicalChannels.toLocaleString()} ` +
    `(${identicalPercentage.toFixed(2)}%)`,
);
console.log();
console.log("Content RGB channels:", contentChannelCount.toLocaleString());
console.log(`Content maximum difference: ${contentMaximumDifference}`);
console.log("Content mean difference: " + contentMeanDifference.toFixed(6));
console.log(
  "Identical content channels: " +
    `${identicalContentChannels.toLocaleString()} ` +
    `(${identicalContentPercentage.toFixed(2)}%)`,
);
console.log(
  "Content differences above 1:",
  contentDifferencesAboveOne.toLocaleString(),
);
console.log(
  "Content differences above 2:",
  contentDifferencesAboveTwo.toLocaleString(),
);
console.log(
  "Content differences above 5:",
  contentDifferencesAboveFive.toLocaleString(),
);
console.log("Padding channels match exactly.");

if (contentMaximumDifference > MAXIMUM_ALLOWED_CONTENT_DIFFERENCE) {
  throw new Error(
    "Content maximum difference exceeded the allowed limit: " +
      `${contentMaximumDifference}`,
  );
}

if (contentMeanDifference > MAXIMUM_ALLOWED_CONTENT_MEAN_DIFFERENCE) {
  throw new Error(
    "Content mean difference exceeded the allowed limit: " +
      contentMeanDifference.toFixed(6),
  );
}

console.log("Breed crop preprocessing parity passed.");
