import { useCallback, useEffect, useMemo, useState, useRef } from "react";
import * as FileSystem from "expo-file-system/legacy";
import { useFocusEffect, useRouter } from "expo-router";
import { useResizer } from "react-native-vision-camera-resizer";
import { createSynchronizable, scheduleOnRN } from "react-native-worklets";
import {
  Image,
  Linking,
  StyleSheet,
  Pressable,
  Text,
  View,
  type LayoutChangeEvent,
} from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameOutput,
  useOrientation,
  usePhotoOutput,
} from "react-native-vision-camera";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  LiveBreedOverlay,
  LIVE_BREED_RETRY_DELAY_UPDATES,
  MAXIMUM_LIVE_BREED_CLASSIFICATION_ATTEMPTS,
  MINIMUM_LIVE_BREED_CONFIDENCE,
  setCapturedPhoto,
} from "@/features/camera";
import {
  decodeMobileDetectorOutput,
  mapDetectorDetectionsToFrame,
  mapFrameDetectionsToPreview,
  suppressDuplicateDetections,
  rotateFrameDetectionsToOrientation,
  createLiveDetectionTrackerState,
  stabilizeLiveFrameDetections,
  BREED_CLASSIFIER_INPUT_BYTE_LENGTH,
  BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH,
  createBreedClassifierInput,
  decodeMobileBreedClassifierOutput,
  findBreedClassificationDetection,
  useMobileInferenceModels,
  type LiveBreedPrediction,
  type LiveFrameDetectionResult,
  type LiveBreedClassificationRequest,
} from "@/features/inference";

import breedLabels from "../generated-assets/models/labels.json";

interface LiveBreedClassificationResult {
  requestId: number;
  trackId: number;
  prediction: LiveBreedPrediction | null;
  errorMessage: string | null;
}

interface CachedLiveBreedPrediction extends LiveBreedPrediction {
  label: string;
}

interface LiveBreedRetryState {
  classificationAttempts: number;
  retryAfterDetectionUpdate: number;
}

//// consts
const LIVE_FRAME_RESOLUTION = {
  width: 640,
  height: 480,
};

const DETECTOR_INPUT_SIZE = 544;

// Throttle inference without queueing frames.
const LIVE_INFERENCE_INTERVAL_MS = 300;

// Classify at most one unclassified track during this interval.
const BREED_CLASSIFICATION_INTERVAL_MS = 400;

const DETECTOR_INPUT_BYTE_LENGTH =
  DETECTOR_INPUT_SIZE *
  DETECTOR_INPUT_SIZE *
  3 *
  Float32Array.BYTES_PER_ELEMENT;

const DETECTOR_OUTPUT_ROWS = 300;
const DETECTOR_VALUES_PER_ROW = 6;
const DETECTOR_OUTPUT_BYTE_LENGTH =
  DETECTOR_OUTPUT_ROWS *
  DETECTOR_VALUES_PER_ROW *
  Float32Array.BYTES_PER_ELEMENT;

/// funcs
export default function CameraScreen() {
  const router = useRouter();
  const { mobileDetector, mobileBreedClassifier } = useMobileInferenceModels();
  const { hasPermission, canRequestPermission, requestPermission } =
    useCameraPermission();
  const backCamera = useCameraDevice("back");

  // Device orientation keeps inference pixels upright. Interface orientation
  // represents the React layout, which may remain portrait when rotation is locked.
  const deviceOrientation = useOrientation("device");
  const interfaceOrientation = useOrientation("interface");

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [liveDetectionResult, setLiveDetectionResult] =
    useState<LiveFrameDetectionResult | null>(null);
  // Tracking history persists between inference updates without causing renders.
  const liveDetectionTracker = useRef(createLiveDetectionTrackerState());
  const [cameraPreviewSize, setCameraPreviewSize] = useState({
    width: 0,
    height: 0,
  });

  // Classification is queued on React, consumed on the Worklet, and cached by
  // stable track ID so a dog is not reclassified on every detector update.
  const breedPredictionsByTrackId = useRef<
    Record<number, CachedLiveBreedPrediction>
  >({});
  const [displayedBreedPredictionsByTrackId, setDisplayedBreedPredictions] =
    useState<Record<number, CachedLiveBreedPrediction>>({});

  // Retry state is also keyed by stable track ID, so each visible dog has its
  // own attempt count and retry delay.
  const breedRetryStateByTrackId = useRef<Record<number, LiveBreedRetryState>>(
    {},
  );
  const liveDetectionUpdateCount = useRef(0);

  const failedBreedTrackIds = useRef(new Set<number>());

  const pendingBreedClassificationRequest =
    useRef<LiveBreedClassificationRequest | null>(null);

  const nextBreedClassificationRequestId = useRef(1);

  const breedClassificationRequest = useMemo(
    () => createSynchronizable<LiveBreedClassificationRequest | null>(null),
    [],
  );

  const lastBreedClassificationTime = useMemo(
    () => createSynchronizable(0),
    [],
  );

  const lastProcessedBreedRequestId = useMemo(
    () => createSynchronizable(0),
    [],
  );

  // Stabilization runs here after the Worklet transfers its small decoded result.
  const handleLiveDetectionResult = useCallback(
    (result: LiveFrameDetectionResult) => {
      liveDetectionUpdateCount.current += 1;

      const trackerUpdate = stabilizeLiveFrameDetections(
        liveDetectionTracker.current,
        result,
      );

      liveDetectionTracker.current = trackerUpdate.state;
      setLiveDetectionResult(trackerUpdate.result);

      const trackedDetections = trackerUpdate.result.detections;
      const activeTrackIds = new Set<number>();

      for (const detection of trackedDetections) {
        if (detection.trackId !== undefined) {
          activeTrackIds.add(detection.trackId);
        }
      }

      // Remove cached results after their tracks have actually expired.
      for (const cachedTrackId of Object.keys(
        breedPredictionsByTrackId.current,
      )) {
        const trackId = Number(cachedTrackId);

        if (!activeTrackIds.has(trackId)) {
          delete breedPredictionsByTrackId.current[trackId];
        }
      }

      for (const retryTrackId of Object.keys(
        breedRetryStateByTrackId.current,
      )) {
        const trackId = Number(retryTrackId);

        if (!activeTrackIds.has(trackId)) {
          delete breedRetryStateByTrackId.current[trackId];
        }
      }

      for (const failedTrackId of failedBreedTrackIds.current) {
        if (!activeTrackIds.has(failedTrackId)) {
          failedBreedTrackIds.current.delete(failedTrackId);
        }
      }

      setDisplayedBreedPredictions({
        ...breedPredictionsByTrackId.current,
      });

      const pendingRequest = pendingBreedClassificationRequest.current;

      if (pendingRequest !== null) {
        const currentPendingDetection = trackedDetections.find(
          (detection) => detection.trackId === pendingRequest.trackId,
        );

        if (currentPendingDetection === undefined) {
          pendingBreedClassificationRequest.current = null;
          breedClassificationRequest.setBlocking(null);
        } else {
          // Keep the pending request aligned with the track's latest box.
          const updatedRequest: LiveBreedClassificationRequest = {
            ...pendingRequest,
            detectorBox: currentPendingDetection.detectorBox,
          };

          pendingBreedClassificationRequest.current = updatedRequest;
          breedClassificationRequest.setBlocking(updatedRequest);
        }
      }

      if (pendingBreedClassificationRequest.current !== null) {
        return;
      }

      // Every visible dog receives its first classification before uncertain
      // predictions are allowed to consume another classifier pass.
      const unclassifiedDetection = trackedDetections.find((detection) => {
        const trackId = detection.trackId;

        return (
          trackId !== undefined &&
          breedPredictionsByTrackId.current[trackId] === undefined &&
          !failedBreedTrackIds.current.has(trackId)
        );
      });

      const retryDetection =
        unclassifiedDetection === undefined
          ? trackedDetections.find((detection) => {
              const trackId = detection.trackId;

              if (
                trackId === undefined ||
                failedBreedTrackIds.current.has(trackId)
              ) {
                return false;
              }

              const cachedPrediction =
                breedPredictionsByTrackId.current[trackId];
              const retryState = breedRetryStateByTrackId.current[trackId];

              return (
                cachedPrediction !== undefined &&
                cachedPrediction.confidence < MINIMUM_LIVE_BREED_CONFIDENCE &&
                retryState !== undefined &&
                retryState.classificationAttempts <
                  MAXIMUM_LIVE_BREED_CLASSIFICATION_ATTEMPTS &&
                liveDetectionUpdateCount.current >=
                  retryState.retryAfterDetectionUpdate
              );
            })
          : undefined;

      const nextDetection = unclassifiedDetection ?? retryDetection;

      if (nextDetection?.trackId === undefined) {
        return;
      }

      const previousRetryState =
        breedRetryStateByTrackId.current[nextDetection.trackId];

      breedRetryStateByTrackId.current[nextDetection.trackId] = {
        classificationAttempts:
          (previousRetryState?.classificationAttempts ?? 0) + 1,

        // The result handler unlocks a later attempt only if the best
        // prediction remains uncertain.
        retryAfterDetectionUpdate: Number.POSITIVE_INFINITY,
      };

      const request: LiveBreedClassificationRequest = {
        requestId: nextBreedClassificationRequestId.current,
        trackId: nextDetection.trackId,
        detectorBox: nextDetection.detectorBox,
      };

      nextBreedClassificationRequestId.current += 1;
      pendingBreedClassificationRequest.current = request;
      breedClassificationRequest.setBlocking(request);
    },
    [breedClassificationRequest],
  );

  const handleLiveBreedClassificationResult = useCallback(
    (result: LiveBreedClassificationResult) => {
      const pendingRequest = pendingBreedClassificationRequest.current;

      // Ignore a stale result produced before an orientation or route reset.
      if (
        pendingRequest === null ||
        pendingRequest.requestId !== result.requestId ||
        pendingRequest.trackId !== result.trackId
      ) {
        return;
      }

      pendingBreedClassificationRequest.current = null;
      breedClassificationRequest.setBlocking(null);

      if (result.prediction === null) {
        failedBreedTrackIds.current.add(result.trackId);

        console.error(
          "Live breed classification failed:",
          result.errorMessage ?? "Unknown classifier error.",
        );

        return;
      }

      const candidatePrediction: CachedLiveBreedPrediction = {
        ...result.prediction,
        label: breedLabels[result.prediction.classId] ?? "Unknown breed",
      };

      const cachedPrediction =
        breedPredictionsByTrackId.current[result.trackId];

      // A retry must not replace a stronger earlier prediction.
      const bestPrediction =
        cachedPrediction !== undefined &&
        cachedPrediction.confidence >= candidatePrediction.confidence
          ? cachedPrediction
          : candidatePrediction;

      breedPredictionsByTrackId.current[result.trackId] = bestPrediction;
      setDisplayedBreedPredictions({
        ...breedPredictionsByTrackId.current,
      });

      const currentRetryState =
        breedRetryStateByTrackId.current[result.trackId];
      const classificationAttempts =
        currentRetryState?.classificationAttempts ?? 1;
      const shouldRetry =
        bestPrediction.confidence < MINIMUM_LIVE_BREED_CONFIDENCE &&
        classificationAttempts < MAXIMUM_LIVE_BREED_CLASSIFICATION_ATTEMPTS;

      breedRetryStateByTrackId.current[result.trackId] = {
        classificationAttempts,

        // Eligibility does not override first-pass priority for new dogs.
        retryAfterDetectionUpdate: shouldRetry
          ? liveDetectionUpdateCount.current + LIVE_BREED_RETRY_DELAY_UPDATES
          : Number.POSITIVE_INFINITY,
      };
    },
    [breedClassificationRequest],
  );

  // A rotated frame uses a different coordinate system, invalidating every
  // tracked box, pending crop request, and cached track-to-breed association.
  useEffect(() => {
    liveDetectionTracker.current = createLiveDetectionTrackerState();
    breedPredictionsByTrackId.current = {};
    breedRetryStateByTrackId.current = {};
    liveDetectionUpdateCount.current = 0;
    failedBreedTrackIds.current.clear();
    pendingBreedClassificationRequest.current = null;

    breedClassificationRequest.setBlocking(null);
    lastBreedClassificationTime.setBlocking(0);
  }, [
    breedClassificationRequest,
    deviceOrientation,
    interfaceOrientation,
    lastBreedClassificationTime,
  ]);

  // React Native reports the final preview size after percentage-based layout
  // has been resolved into physical screen coordinates.
  const handleCameraViewLayout = useCallback(
    ({
      nativeEvent: {
        layout: { width, height },
      },
    }: LayoutChangeEvent) => {
      setCameraPreviewSize((previousSize) => {
        if (previousSize.width === width && previousSize.height === height) {
          return previousSize;
        }

        return { width, height };
      });
    },
    [],
  );

  const photoOutput = usePhotoOutput({
    containerFormat: "jpeg",
    quality: 0.9,
    qualityPrioritization: "quality",
  });

  const detectorModel = mobileDetector.model;
  const breedClassifierModel = mobileBreedClassifier.model;

  // Produce the detector's planar RGB tensor directly from the native frame.
  const liveFrameResizer = useResizer({
    width: DETECTOR_INPUT_SIZE,
    height: DETECTOR_INPUT_SIZE,
    channelOrder: "rgb",
    dataType: "float32",
    pixelLayout: "planar",

    scaleMode: "contain",
  });

  const lastPreprocessTime = useMemo(() => createSynchronizable(0), []);

  const frameResizer = liveFrameResizer.resizer;

  // Captures all frames but only processes up to 4 FPS.
  const frameOutput = useFrameOutput({
    targetResolution: LIVE_FRAME_RESOLUTION,
    pixelFormat: "yuv",
    dropFramesWhileBusy: true,

    // The resizer does not apply Frame.orientation metadata. Rotate the native
    // buffer first so model pixels, detector boxes, and the preview are upright.
    enablePhysicalBufferRotation: true,

    onFrame(frame) {
      "worklet";

      try {
        // Frames are discarded until both asynchronous native resources are ready.
        if (frameResizer == null || detectorModel == null) {
          return;
        }

        const currentTime = performance.now();
        const previousPreprocessTime = lastPreprocessTime.getBlocking();

        if (currentTime - previousPreprocessTime < LIVE_INFERENCE_INTERVAL_MS) {
          return;
        }

        lastPreprocessTime.setBlocking(currentTime);

        const resizedFrame = frameResizer.resize(frame);

        try {
          const detectorInput = resizedFrame.getPixelBuffer();

          // Reject an unexpected native tensor before it reaches the model.
          if (detectorInput.byteLength !== DETECTOR_INPUT_BYTE_LENGTH) {
            throw new Error(
              `Expected ${DETECTOR_INPUT_BYTE_LENGTH} detector input bytes, ` +
                `received ${detectorInput.byteLength}.`,
            );
          }

          // runSync keeps detectorInput valid until native inference has consumed
          // it. This runs on the Worklet thread, not React's UI/JS thread.
          const detectorOutputs = detectorModel.runSync([detectorInput]);

          if (detectorOutputs.length !== 1) {
            throw new Error(
              `Expected one detector output, received ${detectorOutputs.length}.`,
            );
          }

          const detectorOutput = detectorOutputs[0];

          if (
            detectorOutput == null ||
            detectorOutput.byteLength !== DETECTOR_OUTPUT_BYTE_LENGTH
          ) {
            throw new Error(
              `Expected ${DETECTOR_OUTPUT_BYTE_LENGTH} output bytes, received ${
                detectorOutput?.byteLength ?? 0
              }.`,
            );
          }

          // Decode model rows and remove boxes that describe the same physical dog.
          const detectorSpaceDetections = suppressDuplicateDetections(
            decodeMobileDetectorOutput(detectorOutput),
          );

          const classificationRequest =
            breedClassificationRequest.getBlocking();

          const previousBreedClassificationTime =
            lastBreedClassificationTime.getBlocking();

          const previousProcessedRequestId =
            lastProcessedBreedRequestId.getBlocking();

          if (
            breedClassifierModel != null &&
            classificationRequest !== null &&
            classificationRequest.requestId > previousProcessedRequestId &&
            currentTime - previousBreedClassificationTime >=
              BREED_CLASSIFICATION_INTERVAL_MS
          ) {
            const selectedDetection = findBreedClassificationDetection(
              detectorSpaceDetections,
              classificationRequest,
            );

            // If the requested dog moved too far, keep the request pending. React will
            // update its detectorBox when the latest tracked detections arrive.
            if (selectedDetection !== null) {
              lastProcessedBreedRequestId.setBlocking(
                classificationRequest.requestId,
              );

              lastBreedClassificationTime.setBlocking(currentTime);

              try {
                const classifierInput = createBreedClassifierInput(
                  detectorInput,
                  selectedDetection.box,
                );

                if (
                  classifierInput.byteLength !==
                  BREED_CLASSIFIER_INPUT_BYTE_LENGTH
                ) {
                  throw new Error(
                    `Expected ${BREED_CLASSIFIER_INPUT_BYTE_LENGTH} classifier ` +
                      `input bytes, received ${classifierInput.byteLength}.`,
                  );
                }

                const classifierOutputs = breedClassifierModel.runSync([
                  classifierInput,
                ]);

                if (classifierOutputs.length !== 1) {
                  throw new Error(
                    `Expected one classifier output, received ` +
                      `${classifierOutputs.length}.`,
                  );
                }

                const classifierOutput = classifierOutputs[0];

                if (
                  classifierOutput == null ||
                  classifierOutput.byteLength !==
                    BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH
                ) {
                  throw new Error(
                    `Expected ${BREED_CLASSIFIER_OUTPUT_BYTE_LENGTH} classifier ` +
                      `output bytes, received ` +
                      `${classifierOutput?.byteLength ?? 0}.`,
                  );
                }

                const prediction =
                  decodeMobileBreedClassifierOutput(classifierOutput);

                scheduleOnRN(handleLiveBreedClassificationResult, {
                  requestId: classificationRequest.requestId,
                  trackId: classificationRequest.trackId,
                  prediction,
                  errorMessage: null,
                });
              } catch (error) {
                scheduleOnRN(handleLiveBreedClassificationResult, {
                  requestId: classificationRequest.requestId,
                  trackId: classificationRequest.trackId,
                  prediction: null,
                  errorMessage:
                    error instanceof Error ? error.message : String(error),
                });
              }
            }
          }

          // Undo the resizer's square letterboxing. These boxes now refer to the
          // correctly oriented camera frame, but not yet the on-screen preview.
          const frameDetectionResult = mapDetectorDetectionsToFrame(
            detectorSpaceDetections,
            frame.width,
            frame.height,
            frame.orientation,
          );

          // Transfer only small decoded objects across the thread boundary. Camera pixels
          // and model buffers remain native and never enter React state.
          scheduleOnRN(handleLiveDetectionResult, frameDetectionResult);
        } finally {
          // The resized GPU allocation is separate from the original camera frame.
          resizedFrame.dispose();
        }
      } catch (error) {
        console.error("Live detector pipeline failed:", error);
      } finally {
        // This also runs for skipped frames and early returns.
        frame.dispose();
      }
    },
  });

  // VisionCamera can drive the full-resolution photo path and lightweight live
  // frame path from the same native camera session.
  const cameraOutputs = useMemo(
    () => [photoOutput, frameOutput],
    [photoOutput, frameOutput],
  );

  // Successful loading is reflected in the UI; only failures need console output.
  useEffect(() => {
    if (mobileDetector.state === "error") {
      console.error("Mobile detector unavailable:", mobileDetector.error);
    }
  }, [mobileDetector]);

  useEffect(() => {
    if (mobileBreedClassifier.state === "error") {
      console.error(
        "Mobile breed classifier unavailable:",
        mobileBreedClassifier.error,
      );
    }
  }, [mobileBreedClassifier]);

  useEffect(() => {
    if (liveFrameResizer.state === "error") {
      console.error("Live frame resizer unavailable:", liveFrameResizer.error);
    }
  }, [liveFrameResizer]);

  // Stop the native camera session while another route is in front of this one.
  useFocusEffect(
    useCallback(() => {
      setIsCameraActive(true);

      return () => {
        setIsCameraActive(false);
        setIsCameraReady(false);
        setLiveDetectionResult(null);
        liveDetectionTracker.current = createLiveDetectionTrackerState();
        breedPredictionsByTrackId.current = {};
        breedRetryStateByTrackId.current = {};
        liveDetectionUpdateCount.current = 0;
        failedBreedTrackIds.current.clear();
        pendingBreedClassificationRequest.current = null;

        breedClassificationRequest.setBlocking(null);
        lastBreedClassificationTime.setBlocking(0);
      };
    }, [breedClassificationRequest, lastBreedClassificationTime]),
  );

  async function handleCameraPermission() {
    if (canRequestPermission) {
      await requestPermission();
      return;
    }

    // Once permission is permanently denied, only system settings can change it.
    await Linking.openSettings();
  }

  async function handleTakePhoto() {
    if (!isCameraReady || isScanning) {
      return;
    }

    setIsScanning(true);

    try {
      // capturePhoto returns a native object whose memory remains owned by the
      // app until photo.dispose() runs in the nested finally block below.
      const photo = await photoOutput.capturePhoto(
        {
          flashMode: "off",
        },
        {},
      );

      try {
        if (!FileSystem.documentDirectory) {
          throw new Error("The app storage is unavailable.");
        }

        const temporaryPhotoPath = await photo.saveToTemporaryFileAsync();
        const temporaryPhotoUri = `file://${temporaryPhotoPath}`;
        const storedPhotoUri = `${FileSystem.documentDirectory}wanchan-capture-${Date.now()}.jpg`;

        // VisionCamera's temporary file may be reclaimed. Copy it into app
        // document storage before navigating to the static scan route.
        await FileSystem.copyAsync({
          from: temporaryPhotoUri,
          to: storedPhotoUri,
        });

        const storedPhotoInfo = await FileSystem.getInfoAsync(storedPhotoUri);

        if (!storedPhotoInfo.exists) {
          throw new Error("The captured photo could not be stored.");
        }

        setCapturedPhoto({
          uri: storedPhotoUri,
          width: photo.width,
          height: photo.height,
        });

        // The scan route prefers server inference and uses these shared mobile
        // models if the request fails or exceeds its timeout.
        router.push("/staticScan");
      } finally {
        // VisionCamera photos retain native memory until explicitly released.
        photo.dispose();
      }
    } catch (error) {
      console.error("Photo capture failed:", error);
    } finally {
      setIsScanning(false);
    }
  }

  const livePreviewDetections = useMemo(() => {
    if (
      liveDetectionResult === null ||
      deviceOrientation == null ||
      interfaceOrientation == null ||
      cameraPreviewSize.width <= 0 ||
      cameraPreviewSize.height <= 0
    ) {
      return [];
    }

    const interfaceDetectionResult = rotateFrameDetectionsToOrientation(
      liveDetectionResult,
      deviceOrientation,
      interfaceOrientation,
    );

    const previewDetections = mapFrameDetectionsToPreview(
      interfaceDetectionResult,
      cameraPreviewSize.width,
      cameraPreviewSize.height,
    ).detections;

    return previewDetections.map((detection) => ({
      ...detection,
      breedPrediction:
        detection.trackId === undefined
          ? null
          : (displayedBreedPredictionsByTrackId[detection.trackId] ?? null),
    }));
  }, [
    cameraPreviewSize,
    deviceOrientation,
    displayedBreedPredictionsByTrackId,
    interfaceOrientation,
    liveDetectionResult,
  ]);

  if (!hasPermission) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.blurCameraView}>
          <Image
            source={require("../assets/camera-placeholder.png")}
            style={styles.placeholderImage}
            resizeMode="cover"
            blurRadius={18}
          />
          <View style={styles.placeholderOverlay} />

          <View style={[styles.corner, styles.topLeft]} />
          <View style={[styles.corner, styles.topRight]} />
          <View style={[styles.corner, styles.bottomLeft]} />
          <View style={[styles.corner, styles.bottomRight]} />

          <View style={styles.unavailableIcon}>
            <View style={styles.cameraTop} />
            <View style={styles.cameraLens} />
            <View style={styles.cameraSlash} />
          </View>

          <Text style={styles.permissionTitle}>Camera unavailable</Text>
          <Text style={styles.permissionMessage}>
            WanChan Beam needs camera access to identify your dog.
          </Text>

          <Pressable
            onPress={() => {
              void handleCameraPermission();
            }}
            style={styles.permissionButton}
          >
            <Text style={styles.permissionButtonText}>
              {canRequestPermission ? "Allow camera" : "Open settings"}
            </Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  // Device discovery can briefly be empty in release builds, so wait for the
  // concrete back-camera device before mounting VisionCamera.
  if (backCamera === undefined) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.blurCameraView}>
          <Image
            source={require("../assets/camera-placeholder.png")}
            style={styles.placeholderImage}
            resizeMode="cover"
            blurRadius={18}
          />
          <View style={styles.placeholderOverlay} />
          <Text style={styles.permissionTitle}>Preparing camera…</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraView} onLayout={handleCameraViewLayout}>
        {/* Both outputs share this preview; adding live detection must not alter
            the existing photo capture flow. */}
        <Camera
          style={styles.camera}
          device={backCamera}
          isActive={isCameraActive}
          outputs={cameraOutputs}
          resizeMode="cover"
          implementationMode="compatible"
          orientationSource="device"
          onStarted={() => {
            setIsCameraReady(true);
          }}
          onStopped={() => {
            setIsCameraReady(false);
          }}
          onError={(error) => {
            setIsCameraReady(false);
            console.error("Camera session failed:", error);
          }}
        />
        {/* label overlay */}
        <LiveBreedOverlay
          detections={livePreviewDetections}
          previewWidth={cameraPreviewSize.width}
          previewHeight={cameraPreviewSize.height}
        />
        <View style={[styles.corner, styles.topLeft]} />
        <View style={[styles.corner, styles.topRight]} />
        <View style={[styles.corner, styles.bottomLeft]} />
        <View style={[styles.corner, styles.bottomRight]} />
        {/* Report both native model readiness and the latest live dog count. */}
        <View style={styles.liveDetectorStatus}>
          <View
            style={[
              styles.liveDetectorStatusDot,
              mobileDetector.state === "loaded" &&
                styles.liveDetectorStatusDotReady,
              mobileDetector.state === "error" &&
                styles.liveDetectorStatusDotError,
            ]}
          />
          <Text style={styles.liveDetectorStatusText}>
            {mobileDetector.state === "loaded"
              ? "Scan ready"
              : mobileDetector.state === "error"
                ? "Scan unavailable"
                : "Loading scanner..."}
          </Text>
        </View>
        <View style={styles.scanControls}>
          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Take a photo"
            disabled={!isCameraReady || isScanning}
            onPress={handleTakePhoto}
            style={({ pressed }) => [
              styles.scanButton,
              (!isCameraReady || isScanning) && styles.scanButtonDisabled,
              pressed && styles.scanButtonPressed,
            ]}
          >
            <Text style={styles.scanButtonText}>
              {isScanning ? "Taking photo..." : "Take Photo"}
            </Text>
          </Pressable>
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFF8EE",
  },
  blurCameraView: {
    position: "relative",
    width: "90%",
    height: "90%",
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: "#FFF8EE",
  },
  camera: {
    flex: 1,
  },
  placeholderImage: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    width: "auto",
    height: "auto",
  },
  placeholderOverlay: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    backgroundColor: "rgba(0, 0, 0, 0.35)",
  },
  cameraView: {
    position: "relative",
    width: "90%",
    height: "90%",
    overflow: "hidden",
    borderRadius: 14,
    backgroundColor: "#062653",
  },
  corner: {
    position: "absolute",
    width: 40,
    height: 40,
    borderColor: "#5C8FB8",
  },
  topLeft: {
    top: 0,
    left: 0,
    borderTopWidth: 4,
    borderLeftWidth: 4,
    borderTopLeftRadius: 12,
  },
  topRight: {
    top: 0,
    right: 0,
    borderTopWidth: 4,
    borderRightWidth: 4,
    borderTopRightRadius: 12,
  },
  bottomLeft: {
    bottom: 0,
    left: 0,
    borderBottomWidth: 4,
    borderLeftWidth: 4,
    borderBottomLeftRadius: 12,
  },
  bottomRight: {
    right: 0,
    bottom: 0,
    borderRightWidth: 4,
    borderBottomWidth: 4,
    borderBottomRightRadius: 12,
  },
  unavailableIcon: {
    width: 72,
    height: 52,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 3,
    borderColor: "#FFF8EE",
    borderRadius: 12,
    marginBottom: 24,
  },
  cameraTop: {
    position: "absolute",
    top: -10,
    width: 28,
    height: 10,
    backgroundColor: "#FFF8EE",
    borderTopLeftRadius: 6,
    borderTopRightRadius: 6,
  },
  cameraLens: {
    width: 22,
    height: 22,
    borderWidth: 3,
    borderColor: "#FFF8EE",
    borderRadius: 999,
  },
  cameraSlash: {
    position: "absolute",
    width: 86,
    height: 4,
    backgroundColor: "#F3A58F",
    borderRadius: 999,
    transform: [{ rotate: "-38deg" }],
  },
  permissionTitle: {
    color: "#FFF8EE",
    fontSize: 21,
    fontWeight: "700",
    marginBottom: 8,
  },
  permissionMessage: {
    color: "#FFF8EE",
    fontSize: 16,
    textAlign: "center",
    marginBottom: 20,
    paddingHorizontal: 32,
  },
  permissionButton: {
    backgroundColor: "#F3A58F",
    borderRadius: 14,
    paddingHorizontal: 24,
    paddingVertical: 14,
  },
  permissionButtonText: {
    color: "#FFF8EE",
    fontSize: 16,
    fontWeight: "700",
  },
  scanControls: {
    position: "absolute",
    right: 0,
    bottom: 24,
    left: 0,
    alignItems: "center",
    gap: 10,
  },
  liveDetectorStatus: {
    position: "absolute",
    top: 16,
    left: 16,
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 999,
    backgroundColor: "rgba(6, 38, 83, 0.78)",
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  liveDetectorStatusDot: {
    width: 9,
    height: 9,
    borderRadius: 999,
    backgroundColor: "#F3C56B",
  },
  liveDetectorStatusDotReady: {
    backgroundColor: "#7BC89C",
  },
  liveDetectorStatusDotError: {
    backgroundColor: "#F3A58F",
  },
  liveDetectorStatusText: {
    color: "#FFF8EE",
    fontSize: 13,
    fontWeight: "700",
  },
  scanButton: {
    minWidth: 120,
    alignItems: "center",
    backgroundColor: "#F3A58F",
    borderRadius: 20,
    paddingHorizontal: 24,
    paddingVertical: 14,
  },
  scanButtonDisabled: {
    opacity: 0.55,
  },
  scanButtonPressed: {
    transform: [{ scale: 0.96 }],
  },
  scanButtonText: {
    color: "#FFF8EE",
    fontSize: 16,
    fontWeight: "700",
  },
});
