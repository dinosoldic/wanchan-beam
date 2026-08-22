import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { after, test } from "node:test";
import { fileURLToPath } from "node:url";

import { buildApp } from "../src/app.js";
import type { DogDetectionResult } from "../src/inference/index.js";

interface MultipartRequest {
  headers: Record<string, string>;
  payload: Buffer;
}

async function createMultipartRequest(
  fieldName: string,
  contents: Buffer,
  filename: string,
  mimeType: string,
): Promise<MultipartRequest> {
  const formData = new FormData();

  formData.append(
    fieldName,
    new Blob([new Uint8Array(contents)], {
      type: mimeType,
    }),
    filename,
  );

  const request = new Request("http://localhost/detect", {
    method: "POST",
    body: formData,
  });

  return {
    headers: Object.fromEntries(request.headers),
    payload: Buffer.from(await request.arrayBuffer()),
  };
}

const samplePath = fileURLToPath(
  new URL("../../ml/data/samples/test-dogs.png", import.meta.url),
);

const sampleImage = await readFile(samplePath);

const app = await buildApp({
  logger: false,
});

after(async () => {
  await app.close();
});

test("POST /detect returns dog detections", async () => {
  const multipart = await createMultipartRequest(
    "image",
    sampleImage,
    "test-dogs.png",
    "image/png",
  );

  const response = await app.inject({
    method: "POST",
    url: "/detect",
    ...multipart,
  });

  assert.equal(response.statusCode, 200);

  const result = response.json() as DogDetectionResult;

  assert.deepEqual(result.image, {
    width: 860,
    height: 449,
  });

  assert.equal(result.detections.length, 6);

  for (const detection of result.detections) {
    assert.equal(detection.breedPredictions.length, 2);

    const [firstBreed, secondBreed] = detection.breedPredictions;

    for (const prediction of detection.breedPredictions) {
      assert.ok(Number.isInteger(prediction.classId));
      assert.ok(prediction.classId >= 0);
      assert.ok(prediction.classId <= 129);
      assert.ok(prediction.label.length > 0);
      assert.ok(prediction.confidence >= 0);
      assert.ok(prediction.confidence <= 1);
    }

    assert.notEqual(firstBreed.classId, secondBreed.classId);
    assert.ok(firstBreed.confidence >= secondBreed.confidence);
  }
});

test("POST /detect rejects an incorrectly named file field", async () => {
  const multipart = await createMultipartRequest(
    "photo",
    sampleImage,
    "test-dogs.png",
    "image/png",
  );

  const response = await app.inject({
    method: "POST",
    url: "/detect",
    ...multipart,
  });

  assert.equal(response.statusCode, 400);
  assert.deepEqual(response.json(), {
    error: 'The uploaded file field must be named "image"',
  });
});

test("POST /detect rejects an unsupported image type", async () => {
  const multipart = await createMultipartRequest(
    "image",
    sampleImage,
    "test-dogs.png",
    "text/plain",
  );

  const response = await app.inject({
    method: "POST",
    url: "/detect",
    ...multipart,
  });

  assert.equal(response.statusCode, 415);
  assert.deepEqual(response.json(), {
    error: "Unsupported image type: text/plain",
  });
});

test("POST /detect rejects invalid image data", async () => {
  const multipart = await createMultipartRequest(
    "image",
    Buffer.from("This is not an image"),
    "fake.png",
    "image/png",
  );

  const response = await app.inject({
    method: "POST",
    url: "/detect",
    ...multipart,
  });

  assert.equal(response.statusCode, 422);
  assert.deepEqual(response.json(), {
    error: "Uploaded file is not a valid supported image",
  });
});
