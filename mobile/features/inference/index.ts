export { decodeMobileDetectorOutput } from "./decodeMobileDetectorOutput";
export type {
  LiveDetectionBox,
  LiveDogDetection,
} from "./decodeMobileDetectorOutput";
export { suppressDuplicateDetections } from "./suppressDuplicateDetections";

export { mapDetectorDetectionsToFrame } from "./mapDetectorDetectionsToFrame";
export type {
  LiveFrameDetectionResult,
  LiveFrameDogDetection,
  LiveFrameSize,
} from "./mapDetectorDetectionsToFrame";

export { mapFrameDetectionsToPreview } from "./mapFrameDetectionsToPreview";
export type {
  LivePreviewDetectionResult,
  LivePreviewDogDetection,
} from "./mapFrameDetectionsToPreview";
export { rotateFrameDetectionsToOrientation } from "./rotateFrameDetectionsToOrientation";

export {
  createLiveDetectionTrackerState,
  stabilizeLiveFrameDetections,
} from "./stabilizeLiveFrameDetections";

export type {
  LiveDetectionTrackerState,
  LiveDetectionTrackerUpdate,
} from "./stabilizeLiveFrameDetections";

export {
  BREED_CLASSIFIER_INPUT_BYTE_LENGTH,
  BREED_CLASSIFIER_INPUT_SIZE,
  BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH,
  BREED_CLASSIFIER_OUTPUT_CLASSES,
  createBreedClassifierInput,
} from "./createBreedClassifierInput";

export { decodeMobileBreedClassifierOutput } from "./decodeMobileBreedClassifierOutput";
export type { LiveBreedPrediction } from "./decodeMobileBreedClassifierOutput";

export { findBreedClassificationDetection } from "./findBreedClassificationDetection";
export type { LiveBreedClassificationRequest } from "./findBreedClassificationDetection";

export {
  MobileInferenceProvider,
  useMobileInferenceModels,
} from "./MobileInferenceProvider";

export { detectDogsLocally } from "./detectDogsLocally";
export type { LocalInferenceModels } from "./detectDogsLocally";
