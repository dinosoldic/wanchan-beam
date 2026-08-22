import type { CameraOrientation } from "react-native-vision-camera";

import type {
  LiveDetectionBox,
  LiveDogDetection,
} from "./decodeMobileDetectorOutput";

const DETECTOR_INPUT_SIZE = 544;

export interface LiveFrameSize {
  width: number;
  height: number;
}

export interface LiveFrameDogDetection {
  classId: 16;
  label: "dog";
  confidence: number;

  // These coordinates refer to the correctly oriented camera frame.
  box: LiveDetectionBox;
}

export interface LiveFrameDetectionResult {
  frame: LiveFrameSize;
  detections: LiveFrameDogDetection[];
}

export function mapDetectorDetectionsToFrame(
  detections: LiveDogDetection[],
  physicalFrameWidth: number,
  physicalFrameHeight: number,
  orientation: CameraOrientation,
): LiveFrameDetectionResult {
  "worklet";

  if (physicalFrameWidth <= 0 || physicalFrameHeight <= 0) {
    throw new Error(
      `Invalid frame size: ${physicalFrameWidth}x${physicalFrameHeight}.`,
    );
  }

  // The GPU resizer counter-rotates frames into their intended presentation.
  // Left/right orientations therefore swap the physical buffer dimensions.
  const swapsDimensions = orientation === "left" || orientation === "right";

  const frameWidth = swapsDimensions ? physicalFrameHeight : physicalFrameWidth;

  const frameHeight = swapsDimensions
    ? physicalFrameWidth
    : physicalFrameHeight;

  const scale = Math.min(
    DETECTOR_INPUT_SIZE / frameWidth,
    DETECTOR_INPUT_SIZE / frameHeight,
  );

  const resizedWidth = frameWidth * scale;
  const resizedHeight = frameHeight * scale;

  const paddingLeft = (DETECTOR_INPUT_SIZE - resizedWidth) / 2;
  const paddingTop = (DETECTOR_INPUT_SIZE - resizedHeight) / 2;

  const mappedDetections: LiveFrameDogDetection[] = [];

  for (const detection of detections) {
    const x1 = Math.min(
      Math.max((detection.box.x1 - paddingLeft) / scale, 0),
      frameWidth,
    );

    const y1 = Math.min(
      Math.max((detection.box.y1 - paddingTop) / scale, 0),
      frameHeight,
    );

    const x2 = Math.min(
      Math.max((detection.box.x2 - paddingLeft) / scale, 0),
      frameWidth,
    );

    const y2 = Math.min(
      Math.max((detection.box.y2 - paddingTop) / scale, 0),
      frameHeight,
    );

    // Discard boxes that existed entirely inside detector padding.
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    mappedDetections.push({
      classId: detection.classId,
      label: detection.label,
      confidence: detection.confidence,
      box: {
        x1,
        y1,
        x2,
        y2,
      },
    });
  }

  return {
    frame: {
      width: frameWidth,
      height: frameHeight,
    },
    detections: mappedDetections,
  };
}
