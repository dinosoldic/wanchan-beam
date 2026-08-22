import type {
  LiveDetectionBox,
  LiveDogDetection,
} from "./decodeMobileDetectorOutput";

const MINIMUM_REQUEST_IOU = 0.25;

export interface LiveBreedClassificationRequest {
  requestId: number;
  trackId: number;

  // Latest 544x544 box recorded for this stable track.
  detectorBox: LiveDetectionBox;
}

function calculateBoxArea(box: LiveDetectionBox): number {
  "worklet";

  return Math.max(0, box.x2 - box.x1) * Math.max(0, box.y2 - box.y1);
}

function calculateBoxIoU(
  firstBox: LiveDetectionBox,
  secondBox: LiveDetectionBox,
): number {
  "worklet";

  const intersectionWidth = Math.max(
    0,
    Math.min(firstBox.x2, secondBox.x2) - Math.max(firstBox.x1, secondBox.x1),
  );

  const intersectionHeight = Math.max(
    0,
    Math.min(firstBox.y2, secondBox.y2) - Math.max(firstBox.y1, secondBox.y1),
  );

  const intersectionArea = intersectionWidth * intersectionHeight;
  const unionArea =
    calculateBoxArea(firstBox) + calculateBoxArea(secondBox) - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

export function findBreedClassificationDetection(
  detections: LiveDogDetection[],
  request: LiveBreedClassificationRequest,
): LiveDogDetection | null {
  "worklet";

  // Raw Worklet detections do not have React's temporal track IDs. Re-associate
  // the queued track with the current 544x544 box before cropping its pixels.
  let bestDetection: LiveDogDetection | null = null;
  let bestIoU = MINIMUM_REQUEST_IOU;

  for (const detection of detections) {
    const overlap = calculateBoxIoU(detection.box, request.detectorBox);

    if (
      overlap >= MINIMUM_REQUEST_IOU &&
      (bestDetection === null || overlap > bestIoU)
    ) {
      bestDetection = detection;
      bestIoU = overlap;
    }
  }

  return bestDetection;
}
