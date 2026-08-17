import { useEffect, useRef, useState } from "react";
import {
  Image,
  StyleSheet,
  Text,
  View,
  Platform,
  Pressable,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation, useRouter } from "expo-router";

import {
  discardCapturedPhoto,
  getCapturedPhoto,
} from "@/features/camera/capturedPhotoStore";
import {
  DetectionOverlay,
  getContainedImageLayout,
  RetryOverlay,
  ScanningOverlay,
  type Size,
  useResultExport,
} from "@/features/results";
import { detectDogs } from "@/services/RemoteInferenceService";
import type { DogDetectionResponse } from "@/types/detection";

const MINIMUM_SCAN_DURATION_MS = 2000;

export default function StaticScanScreen() {
  const router = useRouter();
  const navigation = useNavigation();

  const capturedPhoto = getCapturedPhoto();
  const [detectionResult, setDetectionResult] =
    useState<DogDetectionResponse | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [previewSize, setPreviewSize] = useState<Size>({
    width: 0,
    height: 0,
  });
  const [scanAttempt, setScanAttempt] = useState(0);

  const resultRef = useRef<View>(null);

  const { downloadResult, shareResult } = useResultExport({
    resultRef,
    setError: setScanError,
  });

  useEffect(() => {
    if (!capturedPhoto) {
      return;
    }

    const photo = capturedPhoto;
    let canceled = false;

    async function scanCapturedPhoto() {
      setIsScanning(true);
      setScanError(null);

      const minimumScanDuration = new Promise<void>((resolve) => {
        setTimeout(resolve, MINIMUM_SCAN_DURATION_MS);
      });

      try {
        const [result] = await Promise.all([
          detectDogs(photo.uri),
          minimumScanDuration,
        ]);

        if (!canceled) {
          setDetectionResult(result);
        }
      } catch (error) {
        await minimumScanDuration;

        console.warn("Camera detection failed:", error);

        if (!canceled) {
          setScanError("Scan failed. Please try again.");
        }
      } finally {
        if (!canceled) {
          setIsScanning(false);
        }
      }
    }

    void scanCapturedPhoto();

    return () => {
      canceled = true;
    };
  }, [capturedPhoto, scanAttempt]);

  useEffect(() => {
    const unsubscribe = navigation.addListener("beforeRemove", () => {
      void discardCapturedPhoto().catch((error) => {
        console.error("Captured photo cleanup failed:", error);
      });
    });

    return unsubscribe;
  }, [navigation]);

  const containedImageLayout = getContainedImageLayout(
    previewSize,
    capturedPhoto,
  );

  if (!capturedPhoto) {
    return (
      <SafeAreaView style={styles.container}>
        <Text style={styles.errorText}>The captured photo is unavailable.</Text>
      </SafeAreaView>
    );
  }

  async function handleRetake() {
    try {
      await discardCapturedPhoto();
    } catch (error) {
      console.error("Captured photo cleanup failed:", error);
    } finally {
      router.back();
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <View
        style={styles.previewArea}
        onLayout={(event) => {
          const { width, height } = event.nativeEvent.layout;

          setPreviewSize({ width, height });
        }}
      >
        {containedImageLayout && (
          <View style={[styles.imagePosition, containedImageLayout]}>
            <View
              ref={resultRef}
              collapsable={false}
              style={styles.resultCanvas}
            >
              <Image
                source={{ uri: capturedPhoto.uri }}
                style={styles.previewImage}
                resizeMode="stretch"
                accessibilityLabel="Captured photo"
              />

              {isScanning && (
                <ScanningOverlay imageHeight={containedImageLayout.height} />
              )}

              {detectionResult && (
                <DetectionOverlay
                  result={detectionResult}
                  displayWidth={containedImageLayout.width}
                  displayHeight={containedImageLayout.height}
                />
              )}

              {scanError && !isScanning && !detectionResult && (
                <RetryOverlay
                  onRetry={() => {
                    setScanAttempt((currentAttempt) => currentAttempt + 1);
                  }}
                />
              )}
            </View>
          </View>
        )}
      </View>

      <View style={styles.footer}>
        <View style={styles.messageSlot}>
          {scanError ? (
            <Text style={styles.errorText}>{scanError}</Text>
          ) : detectionResult?.detections.length === 0 ? (
            <Text style={styles.statusText}>No dogs detected</Text>
          ) : null}
        </View>

        <View style={styles.actionSlot}>
          {detectionResult && (
            <View style={styles.completedActions}>
              <Pressable
                onPress={handleRetake}
                accessibilityRole="button"
                accessibilityLabel="Retake photo"
                style={({ pressed }) => [
                  styles.buttonBase,
                  styles.retakeButton,
                  pressed && styles.buttonPressed,
                ]}
              >
                <Text style={styles.buttonText}>Retake</Text>
              </Pressable>

              <Pressable
                onPress={downloadResult}
                accessibilityRole="button"
                accessibilityLabel="Save scanned image"
                style={({ pressed }) => [
                  styles.iconButton,
                  pressed && styles.iconButtonPressed,
                ]}
              >
                <Ionicons name="download-outline" size={27} color="#FFF8EE" />
              </Pressable>

              {Platform.OS !== "web" && (
                <Pressable
                  onPress={shareResult}
                  accessibilityRole="button"
                  accessibilityLabel="Share scanned image"
                  style={({ pressed }) => [
                    styles.iconButton,
                    pressed && styles.iconButtonPressed,
                  ]}
                >
                  <Ionicons
                    name="paper-plane-outline"
                    size={25}
                    color="#FFF8EE"
                  />
                </Pressable>
              )}
            </View>
          )}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    backgroundColor: "#FFF8EE",
    paddingHorizontal: 24,
    paddingTop: 16,
    paddingBottom: 24,
  },
  previewArea: {
    flex: 1,
    position: "relative",
    width: "100%",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    marginBottom: 20,
    borderRadius: 14,
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  imagePosition: {
    position: "absolute",
    overflow: "hidden",
  },
  footer: {
    width: "100%",
    height: 102,
    alignItems: "center",
    gap: 20,
    flexShrink: 0,
  },
  messageSlot: {
    width: "100%",
    height: 40,
    alignItems: "center",
    justifyContent: "center",
  },
  errorText: {
    color: "#B95C4A",
    fontSize: 16,
    textAlign: "center",
  },
  statusText: {
    color: "#062653",
    fontSize: 24,
    textAlign: "center",
  },
  resultCanvas: {
    position: "relative",
    width: "100%",
    height: "100%",
    overflow: "hidden",
    borderRadius: 14,
  },
  completedActions: {
    width: "90%",
    maxWidth: 300,
    flexDirection: "row",
    gap: 10,
  },
  actionSlot: {
    width: "100%",
    height: 54,
    alignItems: "center",
  },
  buttonBase: {
    height: 54,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#F3A58F",
    paddingHorizontal: 24,
  },
  retakeButton: {
    flex: 1,
  },
  buttonPressed: {
    opacity: 0.8,
    backgroundColor: "#E89079",
  },
  buttonText: {
    color: "#FFF8EE",
    fontSize: 18,
    fontWeight: "700",
  },
  iconButton: {
    width: 54,
    height: 54,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#5C8FB8",
  },
  iconButtonPressed: {
    opacity: 0.8,
  },
});
