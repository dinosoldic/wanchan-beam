import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { classifyDetectedDogs, detectDogs } from "../src/inference/index.js";

const EXPECTED_DOG_COUNT = 6;

const samplePath = fileURLToPath(
  new URL("../../ml/data/samples/test-dogs.png", import.meta.url),
);

const image = await readFile(samplePath);

console.time("Dog detection");

const detectionResult = await detectDogs(image);

console.timeEnd("Dog detection");

if (detectionResult.detections.length !== EXPECTED_DOG_COUNT) {
  throw new Error(
    `Expected ${EXPECTED_DOG_COUNT} dogs, ` +
      `detected ${detectionResult.detections.length}`,
  );
}

console.time("Batched crop classification");

const classifiedDetections = await classifyDetectedDogs(
  image,
  detectionResult.detections,
);

console.timeEnd("Batched crop classification");

if (classifiedDetections.length !== detectionResult.detections.length) {
  throw new Error(
    "Classified detection count does not match detector output: " +
      `expected ${detectionResult.detections.length}, ` +
      `got ${classifiedDetections.length}`,
  );
}

classifiedDetections.forEach((detection, index) => {
  const originalDetection = detectionResult.detections[index];

  if (!originalDetection) {
    throw new Error(`Missing original detection ${index}`);
  }

  if (detection.box !== originalDetection.box) {
    throw new Error(`Detection ${index} was matched with the wrong box`);
  }

  if (detection.breedPredictions.length !== 2) {
    throw new Error(`Detection ${index} does not have two breed predictions`);
  }
});

console.table(
  classifiedDetections.map((detection, index) => {
    const [firstBreed, secondBreed] = detection.breedPredictions;

    return {
      dog: index + 1,
      detectorConfidence: detection.confidence.toFixed(3),
      firstBreed: firstBreed.label,
      firstConfidence: firstBreed.confidence.toFixed(6),
      secondBreed: secondBreed.label,
      secondConfidence: secondBreed.confidence.toFixed(6),
    };
  }),
);

console.log("Detector-to-classifier bridge smoke test passed.");
