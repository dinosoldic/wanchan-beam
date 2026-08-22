import assert from "node:assert/strict";
import test from "node:test";

import { suppressDuplicateDetections } from "../src/postprocessing/suppressDuplicateDetections.js";
import type { DogDetection } from "../src/postprocessing/processDetectorOutput.js";

function createDetection(
  confidence: number,
  box: DogDetection["box"],
): DogDetection {
  return {
    classId: 16,
    label: "dog",
    confidence,
    box,
  };
}

test("removes an almost-identical lower-confidence box", () => {
  const lowerConfidence = createDetection(0.6, {
    x1: 104,
    y1: 104,
    x2: 296,
    y2: 296,
  });

  const higherConfidence = createDetection(0.92, {
    x1: 100,
    y1: 100,
    x2: 300,
    y2: 300,
  });

  const result = suppressDuplicateDetections([
    lowerConfidence,
    higherConfidence,
  ]);

  assert.deepEqual(result, [higherConfidence]);
});

test("keeps two overlapping dogs with different centers", () => {
  const firstDog = createDetection(0.92, {
    x1: 100,
    y1: 100,
    x2: 300,
    y2: 300,
  });

  const secondDog = createDetection(0.87, {
    x1: 180,
    y1: 100,
    x2: 380,
    y2: 300,
  });

  const result = suppressDuplicateDetections([firstDog, secondDog]);

  assert.deepEqual(result, [firstDog, secondDog]);
});

test("removes a contained duplicate with three matching edges", () => {
  const largerDetection = createDetection(0.74, {
    x1: 120,
    y1: 100,
    x2: 272,
    y2: 421,
  });

  const tighterDetection = createDetection(0.3, {
    x1: 168,
    y1: 103,
    x2: 272,
    y2: 421,
  });

  const result = suppressDuplicateDetections([
    largerDetection,
    tighterDetection,
  ]);

  assert.deepEqual(result, [largerDetection]);
});
