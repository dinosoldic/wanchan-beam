import type { LiveDetectionBox } from "./decodeMobileDetectorOutput";
import type {
  LiveFrameDetectionResult,
  LiveFrameDogDetection,
  LiveFrameSize,
} from "./mapDetectorDetectionsToFrame";

export interface LivePreviewDogDetection extends Omit<
  LiveFrameDogDetection,
  "box" | "detectorBox"
> {
  // These coordinates refer to the visible camera preview.
  box: LiveDetectionBox;
}

export interface LivePreviewDetectionResult {
  preview: LiveFrameSize;
  detections: LivePreviewDogDetection[];
}

export function mapFrameDetectionsToPreview(
  result: LiveFrameDetectionResult,
  previewWidth: number,
  previewHeight: number,
): LivePreviewDetectionResult {
  if (previewWidth <= 0 || previewHeight <= 0) {
    throw new Error(
      `Invalid camera preview size: ${previewWidth}x${previewHeight}.`,
    );
  }

  // VisionCamera's "cover" mode enlarges the oriented frame until it fills
  // the entire preview. The excess content is cropped equally on each side.
  const scale = Math.max(
    previewWidth / result.frame.width,
    previewHeight / result.frame.height,
  );

  const renderedWidth = result.frame.width * scale;
  const renderedHeight = result.frame.height * scale;

  const offsetX = (previewWidth - renderedWidth) / 2;
  const offsetY = (previewHeight - renderedHeight) / 2;

  const mappedDetections: LivePreviewDogDetection[] = [];

  for (const detection of result.detections) {
    const x1 = Math.min(
      Math.max(detection.box.x1 * scale + offsetX, 0),
      previewWidth,
    );

    const y1 = Math.min(
      Math.max(detection.box.y1 * scale + offsetY, 0),
      previewHeight,
    );

    const x2 = Math.min(
      Math.max(detection.box.x2 * scale + offsetX, 0),
      previewWidth,
    );

    const y2 = Math.min(
      Math.max(detection.box.y2 * scale + offsetY, 0),
      previewHeight,
    );

    // A detection may be completely outside the visible cropped region.
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    mappedDetections.push({
      trackId: detection.trackId,
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
    preview: {
      width: previewWidth,
      height: previewHeight,
    },
    detections: mappedDetections,
  };
}
