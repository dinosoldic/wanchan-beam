import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { loadDetector } from "../src/inference/index.js";
import { preprocessDetectorImage } from "../src/preprocessing/index.js";

const samplePath = fileURLToPath(
  new URL("../../ml/data/samples/test-dogs.png", import.meta.url),
);

const image = await readFile(samplePath);
const { tensor, transform } = await preprocessDetectorImage(image);
const detector = await loadDetector();

console.time("Inference time");

const outputs = await detector.run({
  images: tensor,
});

console.timeEnd("Inference time");

const output = outputs.output0;

if (!output) {
  throw new Error("Detector did not return output0");
}

if (output.type !== "float32") {
  throw new Error(`Expected float32 output, received ${output.type}`);
}

const values = output.data as Float32Array;
const detectedDogs: Array<{
  confidence: number;
  box: [number, number, number, number];
}> = [];

for (let detectionIndex = 0; detectionIndex < 300; detectionIndex += 1) {
  const offset = detectionIndex * 6;

  const confidence = values[offset + 4]!;
  const classId = Math.round(values[offset + 5]!);

  if (classId !== 16 || confidence < 0.15) {
    continue;
  }

  detectedDogs.push({
    confidence,
    box: [
      values[offset]!,
      values[offset + 1]!,
      values[offset + 2]!,
      values[offset + 3]!,
    ],
  });
}

console.log("Input dimensions:", tensor.dims);
console.log("Output dimensions:", output.dims);
console.log("Letterbox transform:", transform);
console.log("Detected dogs:", detectedDogs.length);
console.table(detectedDogs);
