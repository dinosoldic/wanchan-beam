import assert from "node:assert/strict";
import { test } from "node:test";

import * as ort from "onnxruntime-node";

import { processBreedClassifierOutput } from "../src/postprocessing/index.js";

const BREED_COUNT = 130;

const labels = Array.from(
  {
    length: BREED_COUNT,
  },
  (_, classId) => `breed_${classId}`,
);

function assertApproximatelyEqual(
  actual: number,
  expected: number,
  tolerance = 0.0000001,
): void {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `Expected ${actual} to be within ${tolerance} of ${expected}`,
  );
}

test("calculates confidence across all 130 breeds", () => {
  const logits = new Float32Array(BREED_COUNT);

  const output = new ort.Tensor("float32", logits, [1, BREED_COUNT]);

  const predictions = processBreedClassifierOutput(output, labels);

  assert.equal(predictions.length, 1);

  const [topOne, topTwo] = predictions[0]!;

  // All logits tie, so ascending class ID determines the order.
  assert.equal(topOne.classId, 0);
  assert.equal(topOne.label, "breed_0");

  assert.equal(topTwo.classId, 1);
  assert.equal(topTwo.label, "breed_1");

  // Softmax must include all 130 breeds, not only the selected two.
  assertApproximatelyEqual(topOne.confidence, 1 / BREED_COUNT);
  assertApproximatelyEqual(topTwo.confidence, 1 / BREED_COUNT);
});

test("handles very large logits without overflowing", () => {
  const logits = new Float32Array(BREED_COUNT);

  logits[42] = 10_000;
  logits[9] = 9_999;

  const output = new ort.Tensor("float32", logits, [1, BREED_COUNT]);

  const [[topOne, topTwo]] = processBreedClassifierOutput(output, labels);

  assert.equal(topOne.classId, 42);
  assert.equal(topTwo.classId, 9);

  assert.ok(Number.isFinite(topOne.confidence));
  assert.ok(Number.isFinite(topTwo.confidence));

  assertApproximatelyEqual(topOne.confidence, 0.7310585786);
  assertApproximatelyEqual(topTwo.confidence, 0.2689414214);
});

test("keeps predictions associated with their batch rows", () => {
  const logits = new Float32Array(BREED_COUNT * 2);

  logits.fill(-10);

  // First dog.
  logits[3] = 5;
  logits[4] = 4;

  // Second dog.
  const secondBatchOffset = BREED_COUNT;

  logits[secondBatchOffset + 120] = 6;
  logits[secondBatchOffset + 119] = 5;

  const output = new ort.Tensor("float32", logits, [2, BREED_COUNT]);

  const predictions = processBreedClassifierOutput(output, labels);

  assert.deepEqual(
    predictions.map(([topOne, topTwo]) => [topOne.classId, topTwo.classId]),
    [
      [3, 4],
      [120, 119],
    ],
  );
});

test("rejects an incorrect label count", () => {
  const output = new ort.Tensor("float32", new Float32Array(BREED_COUNT), [
    1,
    BREED_COUNT,
  ]);

  assert.throws(
    () => processBreedClassifierOutput(output, labels.slice(0, 129)),
    {
      message: "Unexpected breed-label count: " + "expected 130, got 129",
    },
  );
});

test("rejects a non-float32 output", () => {
  const output = new ort.Tensor("int32", new Int32Array(BREED_COUNT), [
    1,
    BREED_COUNT,
  ]);

  assert.throws(() => processBreedClassifierOutput(output, labels), {
    message: "Expected float32 classifier output, " + "received int32",
  });
});

test("rejects an unexpected output rank", () => {
  const output = new ort.Tensor("float32", new Float32Array(BREED_COUNT), [
    1,
    1,
    BREED_COUNT,
  ]);

  assert.throws(() => processBreedClassifierOutput(output, labels), {
    message:
      "Expected classifier output rank 2, " + "received shape [1, 1, 130]",
  });
});

test("rejects an unexpected breed dimension", () => {
  const output = new ort.Tensor("float32", new Float32Array(129), [1, 129]);

  assert.throws(() => processBreedClassifierOutput(output, labels), {
    message: "Unexpected classifier output shape: [1, 129]",
  });
});

test("rejects non-finite logits", () => {
  const logits = new Float32Array(BREED_COUNT);

  logits[17] = Number.NaN;

  const output = new ort.Tensor("float32", logits, [1, BREED_COUNT]);

  assert.throws(() => processBreedClassifierOutput(output, labels), {
    message:
      "Classifier returned a non-finite logit " + "for batch 0, class 17",
  });
});
