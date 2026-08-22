import type { CameraOrientation } from "react-native-vision-camera";

import type { LiveDetectionBox } from "./decodeMobileDetectorOutput";
import type { LiveFrameDetectionResult } from "./mapDetectorDetectionsToFrame";

type QuarterTurns = 0 | 1 | 2 | 3;

const ORIENTATION_QUARTER_TURNS: Record<CameraOrientation, QuarterTurns> = {
  up: 0,
  right: 1,
  down: 2,
  left: 3,
};

function rotateBoxClockwise(
  box: LiveDetectionBox,
  frameWidth: number,
  frameHeight: number,
  quarterTurns: QuarterTurns,
): LiveDetectionBox {
  switch (quarterTurns) {
    case 0:
      return box;

    case 1:
      return {
        x1: frameHeight - box.y2,
        y1: box.x1,
        x2: frameHeight - box.y1,
        y2: box.x2,
      };

    case 2:
      return {
        x1: frameWidth - box.x2,
        y1: frameHeight - box.y2,
        x2: frameWidth - box.x1,
        y2: frameHeight - box.y1,
      };

    case 3:
      return {
        x1: box.y1,
        y1: frameWidth - box.x2,
        x2: box.y2,
        y2: frameWidth - box.x1,
      };
  }
}

export function rotateFrameDetectionsToOrientation(
  result: LiveFrameDetectionResult,
  sourceOrientation: CameraOrientation,
  targetOrientation: CameraOrientation,
): LiveFrameDetectionResult {
  // A device-oriented frame must be rotated by the difference between the
  // physical device orientation and the currently displayed UI orientation.
  const quarterTurns = ((ORIENTATION_QUARTER_TURNS[sourceOrientation] -
    ORIENTATION_QUARTER_TURNS[targetOrientation] +
    4) %
    4) as QuarterTurns;

  if (quarterTurns === 0) {
    return result;
  }

  const swapsDimensions = quarterTurns === 1 || quarterTurns === 3;

  return {
    frame: {
      width: swapsDimensions ? result.frame.height : result.frame.width,
      height: swapsDimensions ? result.frame.width : result.frame.height,
    },
    detections: result.detections.map((detection) => ({
      ...detection,
      box: rotateBoxClockwise(
        detection.box,
        result.frame.width,
        result.frame.height,
        quarterTurns,
      ),
    })),
  };
}
