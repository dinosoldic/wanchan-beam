import type { DetectionBox, DogDetection } from "./processDetectorOutput.js";

const DUPLICATE_IOU_THRESHOLD = 0.85;
const MAX_CENTER_OFFSET_RATIO = 0.15;
const DUPLICATE_CONTAINMENT_THRESHOLD = 0.98;
const MATCHING_EDGE_OFFSET_RATIO = 0.03;
const MINIMUM_MATCHING_EDGE_COUNT = 3;

function getBoxWidth(box: DetectionBox): number {
  return box.x2 - box.x1;
}

function getBoxHeight(box: DetectionBox): number {
  return box.y2 - box.y1;
}

function getBoxArea(box: DetectionBox): number {
  return getBoxWidth(box) * getBoxHeight(box);
}

function getIntersectionArea(
  firstBox: DetectionBox,
  secondBox: DetectionBox,
): number {
  const intersectionWidth = Math.max(
    0,
    Math.min(firstBox.x2, secondBox.x2) - Math.max(firstBox.x1, secondBox.x1),
  );

  const intersectionHeight = Math.max(
    0,
    Math.min(firstBox.y2, secondBox.y2) - Math.max(firstBox.y1, secondBox.y1),
  );

  return intersectionWidth * intersectionHeight;
}

function calculateIoU(firstBox: DetectionBox, secondBox: DetectionBox): number {
  const intersectionArea = getIntersectionArea(firstBox, secondBox);
  const unionArea =
    getBoxArea(firstBox) + getBoxArea(secondBox) - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

function calculateContainment(
  firstBox: DetectionBox,
  secondBox: DetectionBox,
): number {
  const smallerArea = Math.min(getBoxArea(firstBox), getBoxArea(secondBox));

  if (smallerArea <= 0) {
    return 0;
  }

  return getIntersectionArea(firstBox, secondBox) / smallerArea;
}

function haveSimilarCenters(
  firstBox: DetectionBox,
  secondBox: DetectionBox,
): boolean {
  const firstCenterX = (firstBox.x1 + firstBox.x2) / 2;
  const firstCenterY = (firstBox.y1 + firstBox.y2) / 2;
  const secondCenterX = (secondBox.x1 + secondBox.x2) / 2;
  const secondCenterY = (secondBox.y1 + secondBox.y2) / 2;

  const referenceWidth = Math.min(
    getBoxWidth(firstBox),
    getBoxWidth(secondBox),
  );

  const referenceHeight = Math.min(
    getBoxHeight(firstBox),
    getBoxHeight(secondBox),
  );

  return (
    Math.abs(firstCenterX - secondCenterX) <=
      referenceWidth * MAX_CENTER_OFFSET_RATIO &&
    Math.abs(firstCenterY - secondCenterY) <=
      referenceHeight * MAX_CENTER_OFFSET_RATIO
  );
}

function haveAtLeastThreeMatchingEdges(
  firstBox: DetectionBox,
  secondBox: DetectionBox,
): boolean {
  const horizontalTolerance =
    Math.min(getBoxWidth(firstBox), getBoxWidth(secondBox)) *
    MATCHING_EDGE_OFFSET_RATIO;

  const verticalTolerance =
    Math.min(getBoxHeight(firstBox), getBoxHeight(secondBox)) *
    MATCHING_EDGE_OFFSET_RATIO;

  const matchingEdges = [
    Math.abs(firstBox.x1 - secondBox.x1) <= horizontalTolerance,
    Math.abs(firstBox.x2 - secondBox.x2) <= horizontalTolerance,
    Math.abs(firstBox.y1 - secondBox.y1) <= verticalTolerance,
    Math.abs(firstBox.y2 - secondBox.y2) <= verticalTolerance,
  ];

  return (
    matchingEdges.filter((edgesMatch) => edgesMatch).length >=
    MINIMUM_MATCHING_EDGE_COUNT
  );
}

export function suppressDuplicateDetections(
  detections: DogDetection[],
): DogDetection[] {
  const detectionsByConfidence = [...detections].sort(
    (first, second) => second.confidence - first.confidence,
  );

  const keptDetections: DogDetection[] = [];

  for (const candidate of detectionsByConfidence) {
    const isDuplicate = keptDetections.some((kept) => {
      const hasNearlyIdenticalGeometry =
        calculateIoU(candidate.box, kept.box) >= DUPLICATE_IOU_THRESHOLD &&
        haveSimilarCenters(candidate.box, kept.box);

      const hasContainedDuplicateGeometry =
        calculateContainment(candidate.box, kept.box) >=
          DUPLICATE_CONTAINMENT_THRESHOLD &&
        haveAtLeastThreeMatchingEdges(candidate.box, kept.box);

      return (
        candidate.classId === kept.classId &&
        (hasNearlyIdenticalGeometry || hasContainedDuplicateGeometry)
      );
    });

    if (!isDuplicate) {
      keptDetections.push(candidate);
    }
  }

  return keptDetections;
}
