export {
  discardCapturedPhoto,
  getCapturedPhoto,
  setCapturedPhoto,
} from "./capturedPhotoStore";
export type { CapturedPhoto } from "./capturedPhotoStore";

export { LiveBreedOverlay } from "./LiveBreedOverlay";
export type { LiveBreedOverlayDetection } from "./LiveBreedOverlay";

export {
  LIVE_BREED_RETRY_DELAY_UPDATES,
  MAXIMUM_LIVE_BREED_CLASSIFICATION_ATTEMPTS,
  MINIMUM_LIVE_BREED_CONFIDENCE,
} from "./liveBreedConfig";
