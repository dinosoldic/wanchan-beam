import { useCallback, useEffect, useMemo, useState } from "react";
import * as FileSystem from "expo-file-system/legacy";
import { useFocusEffect, useRouter } from "expo-router";
import { useResizer } from "react-native-vision-camera-resizer";
import { createSynchronizable } from "react-native-worklets";
import { useTensorflowModel } from "react-native-fast-tflite";
import {
  Image,
  Linking,
  StyleSheet,
  Pressable,
  Text,
  View,
} from "react-native";
import {
  Camera,
  useCameraPermission,
  usePhotoOutput,
  useFrameOutput,
} from "react-native-vision-camera";
import { SafeAreaView } from "react-native-safe-area-context";

import { setCapturedPhoto } from "@/features/camera";
import {
  decodeMobileDetectorOutput,
  suppressDuplicateDetections,
  mapDetectorDetectionsToFrame,
} from "@/features/inference";

// The npm prestart/preandroid hooks copy the shared versioned model here so
// Metro can bundle it without crossing the Windows W:/C: drive boundary.
import mobileDetectorAsset from "../generated-assets/models/mobile-detector.tflite";

//// consts
const LIVE_FRAME_RESOLUTION = {
  width: 640,
  height: 480,
};

const DETECTOR_INPUT_SIZE = 544;

const LIVE_PREPROCESS_INTERVAL_MS = 500;

// Decoding runs at the normal 2 FPS inference rate, but debug output crosses
// into the React Native console only once every two seconds.
const LIVE_DETECTION_LOG_INTERVAL_MS = 2_000;

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
  const { hasPermission, canRequestPermission, requestPermission } =
    useCameraPermission();

  const photoOutput = usePhotoOutput({
    containerFormat: "jpeg",
    quality: 0.9,
    qualityPrioritization: "quality",
  });

  // An empty delegate list deliberately starts with the CPU backend. This gives
  // us a reliable baseline before testing Android GPU or NNAPI acceleration.
  const mobileDetector = useTensorflowModel(mobileDetectorAsset, []);
  const detectorModel = mobileDetector.model;

  // resize input for model
  const liveFrameResizer = useResizer({
    width: DETECTOR_INPUT_SIZE,
    height: DETECTOR_INPUT_SIZE,
    channelOrder: "rgb",
    dataType: "float32",
    pixelLayout: "planar",

    scaleMode: "contain",
  });

  const lastPreprocessTime = useMemo(() => createSynchronizable(0), []);
  const hasLoggedDetectorInput = useMemo(() => createSynchronizable(false), []);
  const hasLoggedDetectorOutput = useMemo(
    () => createSynchronizable(false),
    [],
  );
  const lastDetectionLogTime = useMemo(() => createSynchronizable(0), []);

  const frameResizer = liveFrameResizer.resizer;

  // Captures all frames but only processes 2 FPS
  const frameOutput = useFrameOutput({
    targetResolution: LIVE_FRAME_RESOLUTION,
    pixelFormat: "yuv",
    dropFramesWhileBusy: true,

    onFrame(frame) {
      "worklet";

      try {
        // Frames are discarded until both asynchronous native resources are ready.
        if (frameResizer == null || detectorModel == null) {
          return;
        }

        const currentTime = performance.now();
        const previousPreprocessTime = lastPreprocessTime.getBlocking();

        if (
          currentTime - previousPreprocessTime <
          LIVE_PREPROCESS_INTERVAL_MS
        ) {
          return;
        }

        lastPreprocessTime.setBlocking(currentTime);

        const resizedFrame = frameResizer.resize(frame);

        try {
          const detectorInput = resizedFrame.getPixelBuffer();

          // Log the tensor contract once without continuously crossing from the
          // Worklet thread into the React Native logging thread.
          if (!hasLoggedDetectorInput.getBlocking()) {
            hasLoggedDetectorInput.setBlocking(true);

            console.log("Live detector input prepared:", {
              width: resizedFrame.width,
              height: resizedFrame.height,
              channelOrder: resizedFrame.channelOrder,
              dataType: resizedFrame.dataType,
              pixelLayout: resizedFrame.pixelLayout,
              byteLength: detectorInput.byteLength,
              expectedByteLength: DETECTOR_INPUT_BYTE_LENGTH,
            });
          }

          const inferenceStartedAt = performance.now();

          // runSync keeps detectorInput valid until native inference has consumed
          // it. This runs on the Worklet thread, not React's UI/JS thread.
          const detectorOutputs = detectorModel.runSync([detectorInput]);
          const inferenceTimeMs = performance.now() - inferenceStartedAt;

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

          // Undo the resizer's square letterboxing. These boxes now refer to the
          // correctly oriented camera frame, but not yet the on-screen preview.
          const frameDetectionResult = mapDetectorDetectionsToFrame(
            detectorSpaceDetections,
            frame.width,
            frame.height,
            frame.orientation,
          );

          // Log only the first successful inference so console traffic cannot
          // affect subsequent timing or camera smoothness.
          if (!hasLoggedDetectorOutput.getBlocking()) {
            hasLoggedDetectorOutput.setBlocking(true);

            console.log("Live detector inference passed:", {
              inferenceTimeMs,
              outputCount: detectorOutputs.length,
              outputByteLength: detectorOutput.byteLength,
              expectedOutputByteLength: DETECTOR_OUTPUT_BYTE_LENGTH,
            });
          }

          const previousDetectionLogTime = lastDetectionLogTime.getBlocking();

          if (
            currentTime - previousDetectionLogTime >=
            LIVE_DETECTION_LOG_INTERVAL_MS
          ) {
            lastDetectionLogTime.setBlocking(currentTime);

            console.log(
              "Live dog detections:",
              JSON.stringify({
                physicalFrame: {
                  width: frame.width,
                  height: frame.height,
                  orientation: frame.orientation,
                  isMirrored: frame.isMirrored,
                },
                orientedFrame: frameDetectionResult.frame,
                count: frameDetectionResult.detections.length,
                detections: frameDetectionResult.detections,
              }),
            );
          }
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

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

  // Log the tensor contract reported by the phone. It must match the exported
  // detector before camera pixels are ever passed into the model.
  useEffect(() => {
    if (mobileDetector.state === "loaded") {
      console.log("Mobile detector ready:", {
        inputs: mobileDetector.model.inputs,
        outputs: mobileDetector.model.outputs,
      });
    }
  }, [mobileDetector]);

  // Confirm that this phone supports VisionCamera's GPU resize pipeline before
  // attempting to pass camera buffers through it.
  useEffect(() => {
    if (liveFrameResizer.state === "ready") {
      console.log("Live frame resizer ready.");
    } else if (liveFrameResizer.state === "error") {
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
      };
    }, []),
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

        // The scan route reads this stored URI and sends the still image through
        // the existing server-side detector/classifier pipeline.
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

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraView}>
        {/* Both outputs share this preview; adding live detection must not alter
            the existing photo capture flow. */}
        <Camera
          style={styles.camera}
          device="back"
          isActive={isCameraActive}
          outputs={cameraOutputs}
          resizeMode="cover"
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

        <View style={[styles.corner, styles.topLeft]} />
        <View style={[styles.corner, styles.topRight]} />
        <View style={[styles.corner, styles.bottomLeft]} />
        <View style={[styles.corner, styles.bottomRight]} />

        {/* For now this reports model loading only. It will represent active
            live inference once frames are connected to the detector. */}
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
              ? "Live detector ready"
              : mobileDetector.state === "error"
                ? "Live detector unavailable"
                : "Loading live detector..."}
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
