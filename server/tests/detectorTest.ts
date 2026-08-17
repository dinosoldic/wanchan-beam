import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { loadDetector } from "../src/inference/index.js";
import { processDetectorOutput } from "../src/postprocessing/index.js";
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

const detectedDogs = processDetectorOutput(output, transform);

console.log("Input dimensions:", tensor.dims);
console.log("Output dimensions:", output.dims);
console.log("Letterbox transform:", transform);
console.log("Detected dogs:", detectedDogs.length);

console.table(
  detectedDogs.map(({ confidence, box }) => ({
    confidence: confidence.toFixed(3),
    x1: box.x1.toFixed(1),
    y1: box.y1.toFixed(1),
    x2: box.x2.toFixed(1),
    y2: box.y2.toFixed(1),
  })),
);
