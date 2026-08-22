import type { LiveDetectionBox } from "./decodeMobileDetectorOutput";
import type {
  LiveFrameDetectionResult,
  LiveFrameDogDetection,
  LiveFrameSize,
} from "./mapDetectorDetectionsToFrame";

const MINIMUM_MATCH_IOU = 0.25;
const CURRENT_BOX_WEIGHT = 0.65;
const CURRENT_CONFIDENCE_WEIGHT = 0.5;
const REQUIRED_CONSECUTIVE_HITS = 2;
const IMMEDIATE_CONFIRMATION_CONFIDENCE = 0.65;
const MAXIMUM_MISSED_UPDATES = 2;
const MISSED_CONFIDENCE_DECAY = 0.9;

interface LiveDogDetectionTrack extends LiveFrameDogDetection {
  trackId: number;
  consecutiveHits: number;
  missedUpdates: number;
  isConfirmed: boolean;
}

export interface LiveDetectionTrackerState {
  frame: LiveFrameSize | null;
  nextTrackId: number;
  tracks: LiveDogDetectionTrack[];
}

export interface LiveDetectionTrackerUpdate {
  state: LiveDetectionTrackerState;
  result: LiveFrameDetectionResult;
}

export function createLiveDetectionTrackerState(): LiveDetectionTrackerState {
  return {
    frame: null,
    nextTrackId: 1,
    tracks: [],
  };
}

function calculateBoxIoU(
  first: LiveDetectionBox,
  second: LiveDetectionBox,
): number {
  const intersectionWidth = Math.max(
    0,
    Math.min(first.x2, second.x2) - Math.max(first.x1, second.x1),
  );

  const intersectionHeight = Math.max(
    0,
    Math.min(first.y2, second.y2) - Math.max(first.y1, second.y1),
  );

  const intersectionArea = intersectionWidth * intersectionHeight;

  const firstArea = (first.x2 - first.x1) * (first.y2 - first.y1);
  const secondArea = (second.x2 - second.x1) * (second.y2 - second.y1);

  const unionArea = firstArea + secondArea - intersectionArea;

  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

function smoothValue(
  previous: number,
  current: number,
  currentWeight: number,
): number {
  return previous * (1 - currentWeight) + current * currentWeight;
}

function smoothBox(
  previous: LiveDetectionBox,
  current: LiveDetectionBox,
): LiveDetectionBox {
  return {
    x1: smoothValue(previous.x1, current.x1, CURRENT_BOX_WEIGHT),
    y1: smoothValue(previous.y1, current.y1, CURRENT_BOX_WEIGHT),
    x2: smoothValue(previous.x2, current.x2, CURRENT_BOX_WEIGHT),
    y2: smoothValue(previous.y2, current.y2, CURRENT_BOX_WEIGHT),
  };
}

function removeTrackingMetadata(
  track: LiveDogDetectionTrack,
): LiveFrameDogDetection {
  return {
    classId: track.classId,
    label: track.label,
    confidence: track.confidence,
    box: track.box,
  };
}

export function stabilizeLiveFrameDetections(
  previousState: LiveDetectionTrackerState,
  currentResult: LiveFrameDetectionResult,
): LiveDetectionTrackerUpdate {
  // Rotating the output changes its coordinate system, so old tracks cannot
  // safely be matched against the new frame dimensions.
  const frameMatches =
    previousState.frame?.width === currentResult.frame.width &&
    previousState.frame.height === currentResult.frame.height;

  const previousTracks = frameMatches ? previousState.tracks : [];
  const matchedTrackIndexes = new Set<number>();
  const nextTracks: LiveDogDetectionTrack[] = [];

  let nextTrackId = previousState.nextTrackId;

  for (const detection of currentResult.detections) {
    let bestTrackIndex = -1;
    let bestIoU = MINIMUM_MATCH_IOU;

    for (let index = 0; index < previousTracks.length; index += 1) {
      if (matchedTrackIndexes.has(index)) {
        continue;
      }

      const overlap = calculateBoxIoU(previousTracks[index].box, detection.box);

      if (overlap >= bestIoU) {
        bestIoU = overlap;
        bestTrackIndex = index;
      }
    }

    if (bestTrackIndex >= 0) {
      const previousTrack = previousTracks[bestTrackIndex];
      matchedTrackIndexes.add(bestTrackIndex);

      const consecutiveHits =
        previousTrack.missedUpdates === 0
          ? previousTrack.consecutiveHits + 1
          : 1;

      nextTracks.push({
        ...detection,
        trackId: previousTrack.trackId,
        box: smoothBox(previousTrack.box, detection.box),
        confidence: smoothValue(
          previousTrack.confidence,
          detection.confidence,
          CURRENT_CONFIDENCE_WEIGHT,
        ),
        consecutiveHits,
        missedUpdates: 0,
        isConfirmed:
          previousTrack.isConfirmed ||
          consecutiveHits >= REQUIRED_CONSECUTIVE_HITS ||
          detection.confidence >= IMMEDIATE_CONFIRMATION_CONFIDENCE,
      });

      continue;
    }

    nextTracks.push({
      ...detection,
      trackId: nextTrackId,
      consecutiveHits: 1,
      missedUpdates: 0,
      isConfirmed: detection.confidence >= IMMEDIATE_CONFIRMATION_CONFIDENCE,
    });

    nextTrackId += 1;
  }

  // Keep confirmed tracks briefly when one or two inference frames miss them.
  for (let index = 0; index < previousTracks.length; index += 1) {
    const previousTrack = previousTracks[index];

    if (
      matchedTrackIndexes.has(index) ||
      !previousTrack.isConfirmed ||
      previousTrack.missedUpdates >= MAXIMUM_MISSED_UPDATES
    ) {
      continue;
    }

    nextTracks.push({
      ...previousTrack,
      confidence: previousTrack.confidence * MISSED_CONFIDENCE_DECAY,
      consecutiveHits: 0,
      missedUpdates: previousTrack.missedUpdates + 1,
    });
  }

  return {
    state: {
      frame: currentResult.frame,
      nextTrackId,
      tracks: nextTracks,
    },
    result: {
      frame: currentResult.frame,
      detections: nextTracks
        .filter((track) => track.isConfirmed)
        .map(removeTrackingMetadata),
    },
  };
}
