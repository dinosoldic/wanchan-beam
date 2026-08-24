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

  // Assigned after a detection becomes part of a temporal track.
  trackId?: number;

  // Latest 544x544 model-space box used to create a breed crop.
  detectorBox: LiveDetectionBox;

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

  // Left and right orientations swap the physical frame dimensions.
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

    // Discard boxes that only covered detector padding.
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    mappedDetections.push({
      classId: detection.classId,
      label: detection.label,
      confidence: detection.confidence,

      // Keep detector coordinates for the breed crop.
      detectorBox: detection.box,

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
