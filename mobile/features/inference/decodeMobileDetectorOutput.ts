const DOG_CLASS_ID = 16;
const DOG_CONFIDENCE_THRESHOLD = 0.15;

const DETECTOR_INPUT_SIZE = 544;
const DETECTION_COUNT = 300;
const VALUES_PER_DETECTION = 6;
const EXPECTED_VALUE_COUNT = DETECTION_COUNT * VALUES_PER_DETECTION;

export interface LiveDetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface LiveDogDetection {
  classId: 16;
  label: "dog";
  confidence: number;

  // Boxes still use the square detector coordinates.
  box: LiveDetectionBox;
}

export function decodeMobileDetectorOutput(
  outputBuffer: ArrayBuffer,
): LiveDogDetection[] {
  "worklet";

  const values = new Float32Array(outputBuffer);

  if (values.length !== EXPECTED_VALUE_COUNT) {
    throw new Error(
      `Expected ${EXPECTED_VALUE_COUNT} detector values, received ${values.length}.`,
    );
  }

  const detections: LiveDogDetection[] = [];

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

    // Ignore empty rows, other classes, and weak detections.
    if (classId !== DOG_CLASS_ID || confidence < DOG_CONFIDENCE_THRESHOLD) {
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

    // The end-to-end export returns detector-input pixels.
    const x1 = Math.min(Math.max(rawX1, 0), DETECTOR_INPUT_SIZE);
    const y1 = Math.min(Math.max(rawY1, 0), DETECTOR_INPUT_SIZE);
    const x2 = Math.min(Math.max(rawX2, 0), DETECTOR_INPUT_SIZE);
    const y2 = Math.min(Math.max(rawY2, 0), DETECTOR_INPUT_SIZE);

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    detections.push({
      classId: DOG_CLASS_ID,
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
