import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { detectDogs } from "../src/inference/index.js";

const EXPECTED_DOG_COUNT = 6;

const samplePath = fileURLToPath(
  new URL("../../ml/data/samples/test-dogs.png", import.meta.url),
);

const image = await readFile(samplePath);

console.time("End-to-end detection time");

const result = await detectDogs(image);

console.timeEnd("End-to-end detection time");

if (result.detections.length !== EXPECTED_DOG_COUNT) {
  throw new Error(
    `Expected ${EXPECTED_DOG_COUNT} dogs, detected ${result.detections.length}`,
  );
}

console.log("Original image:", result.image);
console.log("Detected dogs:", result.detections.length);

console.table(
  result.detections.map(({ confidence, box }) => ({
    confidence: confidence.toFixed(3),
    x1: box.x1.toFixed(1),
    y1: box.y1.toFixed(1),
    x2: box.x2.toFixed(1),
    y2: box.y2.toFixed(1),
  })),
);
