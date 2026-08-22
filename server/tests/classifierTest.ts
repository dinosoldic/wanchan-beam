import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import { classifyBreeds } from "../src/inference/index.js";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const MAXIMUM_ALLOWED_CONFIDENCE_DIFFERENCE = 0.000001;

const EXPECTED_PREDICTIONS = [
  {
    classId: 5,
    label: "Australian_Shepherd",
  },
  {
    classId: 98,
    label: "collie",
  },
] as const;

const referencePath = fileURLToPath(
  new URL("./fixtures/breed-crop-python-reference.png", import.meta.url),
);

const referenceImage = await readFile(referencePath);

const { data: referencePixels, info } = await sharp(referenceImage)
  .toColourspace("srgb")
  .raw()
  .toBuffer({
    resolveWithObject: true,
  });

if (
  info.width !== INPUT_SIZE ||
  info.height !== INPUT_SIZE ||
  info.channels !== CHANNEL_COUNT
) {
  throw new Error(
    "Unexpected classifier smoke-test input: " +
      `${info.width}x${info.height}x${info.channels}`,
  );
}

console.time("Single-crop breed classification");

const singlePredictions = await classifyBreeds([referencePixels]);

console.timeEnd("Single-crop breed classification");

assert.equal(singlePredictions.length, 1);

const singleDogPredictions = singlePredictions[0]!;

assert.deepEqual(
  singleDogPredictions.map(({ classId, label }) => ({
    classId,
    label,
  })),
  EXPECTED_PREDICTIONS,
);

assert.equal(singleDogPredictions.length, 2);
assert.ok(
  singleDogPredictions[0].confidence >= singleDogPredictions[1].confidence,
);

console.time("Two-crop breed classification");

const batchPredictions = await classifyBreeds([
  referencePixels,
  referencePixels,
]);

console.timeEnd("Two-crop breed classification");

assert.equal(batchPredictions.length, 2);

const firstBatchPredictions = batchPredictions[0]!;
const secondBatchPredictions = batchPredictions[1]!;

// Duplicated images must keep the same ranked IDs and labels.
for (let predictionIndex = 0; predictionIndex < 2; predictionIndex += 1) {
  const singlePrediction = singleDogPredictions[predictionIndex]!;
  const firstBatchPrediction = firstBatchPredictions[predictionIndex]!;
  const secondBatchPrediction = secondBatchPredictions[predictionIndex]!;

  assert.equal(firstBatchPrediction.classId, singlePrediction.classId);
  assert.equal(firstBatchPrediction.label, singlePrediction.label);

  assert.equal(secondBatchPrediction.classId, singlePrediction.classId);
  assert.equal(secondBatchPrediction.label, singlePrediction.label);
}

const confidenceDifferences = [
  ...firstBatchPredictions.map((prediction, index) =>
    Math.abs(prediction.confidence - singleDogPredictions[index]!.confidence),
  ),
  ...secondBatchPredictions.map((prediction, index) =>
    Math.abs(prediction.confidence - singleDogPredictions[index]!.confidence),
  ),
];

const maximumConfidenceDifference = Math.max(...confidenceDifferences);

if (maximumConfidenceDifference > MAXIMUM_ALLOWED_CONFIDENCE_DIFFERENCE) {
  throw new Error(
    "Batched classification confidence changed beyond " +
      "the allowed limit: " +
      maximumConfidenceDifference.toFixed(12),
  );
}

console.table(
  singleDogPredictions.map(({ classId, label, confidence }) => ({
    classId,
    label,
    confidence: confidence.toFixed(6),
  })),
);

console.log(
  "Maximum confidence difference between " + "single and batched inference:",
  maximumConfidenceDifference.toFixed(12),
);

console.log("Breed classifier smoke test passed.");
