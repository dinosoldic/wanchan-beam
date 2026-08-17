import type { Tensor } from "onnxruntime-node";

import type { LetterboxTransform } from "../preprocessing/index.js";

const DOG_CLASS_ID = 16;
const CONFIDENCE_THRESHOLD = 0.15;

const DETECTION_COUNT = 300;
const VALUES_PER_DETECTION = 6;
const EXPECTED_VALUE_COUNT = DETECTION_COUNT * VALUES_PER_DETECTION;

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface DogDetection {
  classId: number;
  label: "dog";
  confidence: number;
  box: DetectionBox;
}

function clampToImage(value: number, maximum: number): number {
  return Math.min(Math.max(value, 0), maximum);
}

export function processDetectorOutput(
  output: Tensor,
  transform: LetterboxTransform,
): DogDetection[] {
  const [batchSize, detectionCount, valuesPerDetection] = output.dims;

  if (
    batchSize !== 1 ||
    detectionCount !== DETECTION_COUNT ||
    valuesPerDetection !== VALUES_PER_DETECTION
  ) {
    throw new Error(
      `Unexpected detector output shape: [${output.dims.join(", ")}]`,
    );
  }

  if (output.type !== "float32") {
    throw new Error(
      `Expected float32 detector output, received ${output.type}`,
    );
  }

  const values = output.data as Float32Array;

  if (values.length !== EXPECTED_VALUE_COUNT) {
    throw new Error(
      `Expected ${EXPECTED_VALUE_COUNT} output values, received ${values.length}`,
    );
  }

  const detections: DogDetection[] = [];

  for (
    let detectionIndex = 0;
    detectionIndex < DETECTION_COUNT;
    detectionIndex += 1
  ) {
    const offset = detectionIndex * VALUES_PER_DETECTION;

    const rawX1 = values[offset]!;
    const rawY1 = values[offset + 1]!;
    const rawX2 = values[offset + 2]!;
    const rawY2 = values[offset + 3]!;
    const confidence = values[offset + 4]!;
    const classId = Math.round(values[offset + 5]!);

    if (classId !== DOG_CLASS_ID || confidence < CONFIDENCE_THRESHOLD) {
      continue;
    }

    if (
      !Number.isFinite(rawX1) ||
      !Number.isFinite(rawY1) ||
      !Number.isFinite(rawX2) ||
      !Number.isFinite(rawY2) ||
      !Number.isFinite(confidence)
    ) {
      continue;
    }

    const x1 = clampToImage(
      (rawX1 - transform.paddingLeft) / transform.scale,
      transform.originalWidth,
    );

    const y1 = clampToImage(
      (rawY1 - transform.paddingTop) / transform.scale,
      transform.originalHeight,
    );

    const x2 = clampToImage(
      (rawX2 - transform.paddingLeft) / transform.scale,
      transform.originalWidth,
    );

    const y2 = clampToImage(
      (rawY2 - transform.paddingTop) / transform.scale,
      transform.originalHeight,
    );

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    detections.push({
      classId,
      label: "dog",
      confidence,
      box: {
        x1,
        y1,
        x2,
        y2,
      },
    });
  }

  return detections;
}
