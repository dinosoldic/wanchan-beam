import type { TfliteModel } from "react-native-fast-tflite";

import breedLabels from "@/generated-assets/models/labels.json";
import type {
  BreedPrediction,
  DogBreedPredictions,
  DogDetection,
  DogDetectionResponse,
} from "@/types/detection";

import {
  BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH,
  createBreedClassifierInput,
} from "./createBreedClassifierInput";
import {
  createStaticDetectorInput,
  type StaticDetectorTransform,
} from "./createStaticDetectorInput";
import { decodeMobileDetectorOutput } from "./decodeMobileDetectorOutput";
import type { LiveDetectionBox } from "./decodeMobileDetectorOutput";
import { suppressDuplicateDetections } from "./suppressDuplicateDetections";

const DETECTOR_OUTPUT_BYTE_LENGTH =
  300 * 6 * Float32Array.BYTES_PER_ELEMENT;

export interface LocalInferenceModels {
  detectorModel: TfliteModel;
  breedClassifierModel: TfliteModel;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function mapDetectorBoxToOriginalImage(
  box: LiveDetectionBox,
  transform: StaticDetectorTransform,
): LiveDetectionBox | null {
  const horizontalScale = transform.originalWidth / transform.resizedWidth;
  const verticalScale = transform.originalHeight / transform.resizedHeight;

  const mappedBox = {
    x1: clamp(
      (box.x1 - transform.paddingLeft) * horizontalScale,
      0,
      transform.originalWidth,
    ),
    y1: clamp(
      (box.y1 - transform.paddingTop) * verticalScale,
      0,
      transform.originalHeight,
    ),
    x2: clamp(
      (box.x2 - transform.paddingLeft) * horizontalScale,
      0,
      transform.originalWidth,
    ),
    y2: clamp(
      (box.y2 - transform.paddingTop) * verticalScale,
      0,
      transform.originalHeight,
    ),
  };

  return mappedBox.x2 > mappedBox.x1 && mappedBox.y2 > mappedBox.y1
    ? mappedBox
    : null;
}

function decodeTopTwoBreedPredictions(
  outputBuffer: ArrayBuffer,
): DogBreedPredictions {
  if (outputBuffer.byteLength !== BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH) {
    throw new Error(
      `Expected ${BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH} classifier output ` +
        `bytes, received ${outputBuffer.byteLength}.`,
    );
  }

  const logits = new Float32Array(outputBuffer);

  let firstClassId = -1;
  let secondClassId = -1;

  for (let classId = 0; classId < logits.length; classId += 1) {
    const logit = logits[classId]!;

    if (!Number.isFinite(logit)) {
      throw new Error(`Breed logit ${classId} is not finite.`);
    }

    if (firstClassId === -1 || logit > logits[firstClassId]!) {
      secondClassId = firstClassId;
      firstClassId = classId;
    } else if (secondClassId === -1 || logit > logits[secondClassId]!) {
      secondClassId = classId;
    }
  }

  if (firstClassId < 0 || secondClassId < 0) {
    throw new Error("Breed classifier did not return two usable classes.");
  }

  const maximumLogit = logits[firstClassId]!;
  let exponentialSum = 0;

  for (const logit of logits) {
    exponentialSum += Math.exp(logit - maximumLogit);
  }

  if (!Number.isFinite(exponentialSum) || exponentialSum <= 0) {
    throw new Error("Breed softmax produced an invalid sum.");
  }

  function createPrediction(classId: number): BreedPrediction {
    return {
      classId,
      label: breedLabels[classId] ?? "Unknown breed",
      confidence: Math.exp(logits[classId]! - maximumLogit) / exponentialSum,
    };
  }

  return [createPrediction(firstClassId), createPrediction(secondClassId)];
}

export async function detectDogsLocally(
  imageUri: string,
  models: LocalInferenceModels,
): Promise<DogDetectionResponse> {
  const { buffer: detectorInput, transform } =
    await createStaticDetectorInput(imageUri);

  const detectorOutputs = await models.detectorModel.run([detectorInput]);

  if (detectorOutputs.length !== 1) {
    throw new Error(
      `Expected one detector output, received ${detectorOutputs.length}.`,
    );
  }

  const detectorOutput = detectorOutputs[0];

  if (
    detectorOutput === undefined ||
    detectorOutput.byteLength !== DETECTOR_OUTPUT_BYTE_LENGTH
  ) {
    throw new Error(
      `Expected ${DETECTOR_OUTPUT_BYTE_LENGTH} detector output bytes, received ` +
        `${detectorOutput?.byteLength ?? 0}.`,
    );
  }

  const detectorSpaceDetections = suppressDuplicateDetections(
    decodeMobileDetectorOutput(detectorOutput),
  );

  const detections: DogDetection[] = [];

  // Static inference has no frame deadline, so classify sequentially to keep
  // device memory bounded when one photo contains many dogs.
  for (const detection of detectorSpaceDetections) {
    const mappedBox = mapDetectorBoxToOriginalImage(detection.box, transform);

    if (mappedBox === null) {
      continue;
    }

    const classifierInput = createBreedClassifierInput(
      detectorInput,
      detection.box,
    );

    const classifierOutputs = await models.breedClassifierModel.run([
      classifierInput,
    ]);

    if (classifierOutputs.length !== 1 || classifierOutputs[0] === undefined) {
      throw new Error(
        `Expected one classifier output, received ${classifierOutputs.length}.`,
      );
    }

    detections.push({
      classId: 16,
      label: "dog",
      confidence: detection.confidence,
      box: mappedBox,
      breedPredictions: decodeTopTwoBreedPredictions(classifierOutputs[0]),
    });
  }

  return {
    image: {
      width: transform.originalWidth,
      height: transform.originalHeight,
    },
    detections,
  };
}
