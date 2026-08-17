import { useState, useRef } from "react";
import { Ionicons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import {
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

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

export default function UploadScreen() {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState<Size | null>(null);
  const [detectionResult, setDetectionResult] =
    useState<DogDetectionResponse | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [previewSize, setPreviewSize] = useState<Size>({
    width: 0,
    height: 0,
  });
  const resultRef = useRef<View>(null);
  const { downloadResult, shareResult } = useResultExport({
    resultRef,
    setError: setScanError,
  });

  async function pickImage() {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["images"],
      allowsEditing: false,
      quality: 0.85,
    });

    if (result.canceled) {
      return;
    }

    const image = result.assets[0];

    if (!image) {
      return;
    }

    setImageUri(image.uri);
    setImageSize({
      width: image.width,
      height: image.height,
    });
    setDetectionResult(null);
    setScanError(null);
  }

  async function scanImage() {
    if (!imageUri || isScanning) {
      return;
    }

    setIsScanning(true);
    setScanError(null);

    const minimumScanDuration = new Promise<void>((resolve) => {
      setTimeout(resolve, MINIMUM_SCAN_DURATION_MS);
    });

    try {
      const [result] = await Promise.all([
        detectDogs(imageUri),
        minimumScanDuration,
      ]);

      setDetectionResult(result);
    } catch (error) {
      await minimumScanDuration;

      console.warn("Detection failed:", error);
      setScanError("Scan failed. Please try again.");
    } finally {
      setIsScanning(false);
    }
  }

  const containedImageLayout = getContainedImageLayout(previewSize, imageSize);

  return (
    <SafeAreaView style={styles.container}>
      <Pressable
        onPress={pickImage}
        disabled={isScanning || detectionResult !== null}
        onLayout={(event) => {
          const { width, height } = event.nativeEvent.layout;

          setPreviewSize({ width, height });
        }}
        accessibilityRole="button"
        accessibilityLabel="Choose a photo"
        style={({ pressed }) => [
          styles.previewArea,
          pressed && styles.previewAreaPressed,
        ]}
      >
        {imageUri && containedImageLayout ? (
          <View style={[styles.resultPosition, containedImageLayout]}>
            <View
              ref={resultRef}
              collapsable={false}
              style={styles.resultCanvas}
            >
              <Image
                source={{ uri: imageUri }}
                style={styles.previewImage}
                resizeMode="stretch"
                accessibilityLabel="Selected photo"
              />

              {detectionResult && (
                <DetectionOverlay
                  result={detectionResult}
                  displayWidth={containedImageLayout.width}
                  displayHeight={containedImageLayout.height}
                />
              )}

              {isScanning && (
                <ScanningOverlay imageHeight={containedImageLayout.height} />
              )}

              {scanError && !isScanning && !detectionResult && (
                <RetryOverlay onRetry={() => void scanImage()} />
              )}
            </View>
          </View>
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderTitle}>Choose a photo</Text>
            <Text style={styles.placeholderMessage}>
              Tap here to select an image from your device
            </Text>
          </View>
        )}
      </Pressable>

      <View style={styles.footer}>
        <View style={styles.messageSlot}>
          {scanError ? (
            <Text style={styles.errorText}>{scanError}</Text>
          ) : detectionResult?.detections.length === 0 ? (
            <Text style={styles.statusText}>No dogs detected</Text>
          ) : null}
        </View>

        <View style={styles.actionSlot}>
          {imageUri &&
            (detectionResult ? (
              <View style={styles.completedActions}>
                <Pressable
                  onPress={pickImage}
                  accessibilityRole="button"
                  accessibilityLabel="Choose a new photo"
                  style={({ pressed }) => [
                    styles.buttonBase,
                    styles.newUploadButton,
                    pressed && styles.buttonPressed,
                  ]}
                >
                  <Text style={styles.buttonText}>New Upload</Text>
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
            ) : scanError ? null : (
              <Pressable
                disabled={isScanning}
                onPress={scanImage}
                accessibilityRole="button"
                accessibilityLabel="Scan the selected photo"
                style={({ pressed }) => [
                  styles.buttonBase,
                  styles.scanButton,
                  isScanning && styles.buttonDisabled,
                  pressed && styles.buttonPressed,
                ]}
              >
                <Text style={styles.buttonText}>
                  {isScanning ? "Scanning..." : "Scan Photo"}
                </Text>
              </Pressable>
            ))}
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
    borderWidth: 2,
    borderStyle: "dashed",
    borderColor: "#5C8FB8",
    borderRadius: 14,
    overflow: "hidden",
    backgroundColor: "#FFFCF7",
    marginBottom: 20,
  },
  previewAreaPressed: {
    opacity: 0.8,
  },
  previewImage: {
    width: "100%",
    height: "100%",
  },
  resultPosition: {
    position: "absolute",
    overflow: "hidden",
  },
  resultCanvas: {
    position: "relative",
    width: "100%",
    height: "100%",
    overflow: "hidden",
  },
  placeholder: {
    alignItems: "center",
    paddingHorizontal: 32,
  },
  placeholderTitle: {
    color: "#062653",
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 8,
  },
  placeholderMessage: {
    color: "#5C8FB8",
    fontSize: 15,
    textAlign: "center",
  },
  buttonBase: {
    height: 54,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#F3A58F",
    paddingHorizontal: 24,
  },
  scanButton: {
    width: "60%",
    maxWidth: 300,
  },
  completedActions: {
    width: "90%",
    maxWidth: 300,
    flexDirection: "row",
    gap: 10,
  },
  newUploadButton: {
    flex: 1,
  },
  buttonPressed: {
    opacity: 0.8,
    backgroundColor: "#E89079",
  },
  buttonDisabled: {
    opacity: 0.55,
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
  footer: {
    width: "100%",
    height: 102,
    alignItems: "center",
    gap: 20,
    flexShrink: 0,
  },
  messageSlot: {
    width: "100%",
    height: 28,
    alignItems: "center",
    justifyContent: "center",
  },
  actionSlot: {
    width: "100%",
    height: 54,
    alignItems: "center",
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
  iconButtonPressed: {
    opacity: 0.8,
  },
});
