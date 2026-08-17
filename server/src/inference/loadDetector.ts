import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-node";

const defaultModelPath = fileURLToPath(
  new URL("../../../models/v1/detector.onnx", import.meta.url),
);

const modelPath = process.env.DETECTOR_MODEL_PATH ?? defaultModelPath;

let sessionPromise: Promise<ort.InferenceSession> | undefined;

export function loadDetector(): Promise<ort.InferenceSession> {
  sessionPromise ??= ort.InferenceSession.create(modelPath, {
    executionProviders: ["cpu"],
    graphOptimizationLevel: "all",
  });

  return sessionPromise;
}
