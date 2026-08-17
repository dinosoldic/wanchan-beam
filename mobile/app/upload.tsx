import { useEffect, useState, useRef } from "react";
import { Ionicons } from "@expo/vector-icons";
import domToImage from "dom-to-image";
import * as ImagePicker from "expo-image-picker";
import * as MediaLibrary from "expo-media-library/legacy";
import { isRunningInExpoGo } from "expo";
import * as Sharing from "expo-sharing";
import { captureRef } from "react-native-view-shot";
import { LinearGradient } from "expo-linear-gradient";
import {
  Alert,
  Animated,
  Easing,
  Image,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { detectDogs } from "@/services/RemoteInferenceService";
import type { DogDetectionResponse } from "@/types/detection";
import { DetectionOverlay } from "@/features/results/DetectionOverlay";

const MINIMUM_SCAN_DURATION_MS = 2000;
const SCAN_TRAVEL_DURATION_MS = 900;
const SCAN_LINE_HEIGHT = 80;

interface Size {
  width: number;
  height: number;
}

interface ContainedImageLayout extends Size {
  top: number;
  left: number;
}

function getContainedImageLayout(
  container: Size,
  image: Size | null,
): ContainedImageLayout | null {
  if (
    !image ||
    container.width <= 0 ||
    container.height <= 0 ||
    image.width <= 0 ||
    image.height <= 0
  ) {
    return null;
  }

  const scale = Math.min(
    container.width / image.width,
    container.height / image.height,
  );
  const width = image.width * scale;
  const height = image.height * scale;

  return {
    width,
    height,
    left: (container.width - width) / 2,
    top: (container.height - height) / 2,
  };
}

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
  const [scanProgress] = useState(() => new Animated.Value(0));
  const resultRef = useRef<View>(null);

  useEffect(() => {
    if (!isScanning) {
      scanProgress.stopAnimation();
      scanProgress.setValue(0);
      return;
    }

    const scanAnimation = Animated.loop(
      Animated.sequence([
        Animated.timing(scanProgress, {
          toValue: 1,
          duration: SCAN_TRAVEL_DURATION_MS,
          easing: Easing.linear,
          useNativeDriver: true,
        }),
        Animated.timing(scanProgress, {
          toValue: 0,
          duration: SCAN_TRAVEL_DURATION_MS,
          easing: Easing.linear,
          useNativeDriver: true,
        }),
      ]),
    );

    scanAnimation.start();

    return () => {
      scanAnimation.stop();
    };
  }, [isScanning, scanProgress]);

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
      console.log("Detection result:", result);
    } catch (error) {
      await minimumScanDuration;

      console.error("Detection failed:", error);
      setScanError("Scan failed. Please try again.");
    } finally {
      setIsScanning(false);
    }
  }

  async function downloadResult() {
    if (!resultRef.current || !detectionResult) {
      return;
    }

    setScanError(null);

    try {
      if (Platform.OS === "web") {
        const dataUrl = await domToImage.toPng(
          resultRef.current as unknown as Node,
          {
            quality: 1,
          },
        );

        const downloadLink = document.createElement("a");

        downloadLink.download = `wanchan-beam-${Date.now()}.png`;
        downloadLink.href = dataUrl;

        document.body.appendChild(downloadLink);
        downloadLink.click();
        downloadLink.remove();

        return;
      }

      if (isRunningInExpoGo()) {
        setScanError("Photo access is required to save the scanned image.");
        return;
      }

      const localUri = await captureRef(resultRef, {
        format: "png",
        quality: 1,
        result: "tmpfile",
      });

      const permission = await MediaLibrary.requestPermissionsAsync(true, [
        "photo",
      ]);

      if (!permission.granted) {
        setScanError("Photo access is required to save the scanned image.");
        return;
      }

      await MediaLibrary.saveToLibraryAsync(localUri);

      Alert.alert(
        "Image saved",
        "The scanned image was saved to your photo library.",
      );
    } catch (error) {
      console.error("Download failed:", error);
      setScanError("Could not save the scanned image.");
    }
  }

  async function shareResult() {
    if (Platform.OS === "web" || !resultRef.current || !detectionResult) {
      return;
    }

    setScanError(null);

    try {
      const sharingAvailable = await Sharing.isAvailableAsync();

      if (!sharingAvailable) {
        setScanError("Sharing is not available on this device.");
        return;
      }

      const localUri = await captureRef(resultRef, {
        format: "png",
        quality: 1,
        result: "tmpfile",
      });

      await Sharing.shareAsync(localUri, {
        mimeType: "image/png",
        dialogTitle: "Share scanned image",
        UTI: "public.png",
      });
    } catch (error) {
      console.error("Sharing failed:", error);
      setScanError("Could not share the scanned image.");
    }
  }

  const containedImageLayout = getContainedImageLayout(previewSize, imageSize);
  const scanLinePosition = scanProgress.interpolate({
    inputRange: [0, 1],
    outputRange: [
      0,
      Math.max((containedImageLayout?.height ?? 0) - SCAN_LINE_HEIGHT, 0),
    ],
  });

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
                <View style={styles.scanningOverlay} pointerEvents="none">
                  <Animated.View
                    style={[
                      styles.scanLine,
                      {
                        transform: [{ translateY: scanLinePosition }],
                      },
                    ]}
                  >
                    <LinearGradient
                      colors={[
                        "rgba(92, 143, 184, 0)",
                        "rgba(92, 143, 184, 0.12)",
                        "rgba(92, 143, 184, 0.34)",
                        "rgba(92, 143, 184, 0.12)",
                        "rgba(92, 143, 184, 0)",
                      ]}
                      locations={[0, 0.25, 0.5, 0.75, 1]}
                      start={{ x: 0.5, y: 0 }}
                      end={{ x: 0.5, y: 1 }}
                      style={styles.scanBandGradient}
                    />

                    <LinearGradient
                      colors={[
                        "rgba(92, 143, 184, 0)",
                        "rgba(92, 143, 184, 0.75)",
                        "rgba(92, 143, 184, 1)",
                        "rgba(92, 143, 184, 0.75)",
                        "rgba(92, 143, 184, 0)",
                      ]}
                      locations={[0, 0.2, 0.5, 0.8, 1]}
                      start={{ x: 0, y: 0.5 }}
                      end={{ x: 1, y: 0.5 }}
                      style={styles.scanBeam}
                    />
                  </Animated.View>
                </View>
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

      {scanError && <Text style={styles.errorText}>{scanError}</Text>}

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
                styles.downloadButton,
                pressed && styles.downloadButtonPressed,
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
                  styles.shareButton,
                  pressed && styles.downloadButtonPressed,
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
        ) : (
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
  downloadButton: {
    width: 54,
    height: 54,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#5C8FB8",
  },
  shareButton: {
    width: 54,
    height: 54,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 10,
    backgroundColor: "#5C8FB8",
  },
  scanningOverlay: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    overflow: "hidden",
  },
  scanLine: {
    position: "absolute",
    top: 0,
    right: 0,
    left: 0,
    height: SCAN_LINE_HEIGHT,
  },
  scanBandGradient: {
    position: "absolute",
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },
  scanBeam: {
    position: "absolute",
    top: "50%",
    right: "4%",
    left: "4%",
    height: 4,
    borderRadius: 999,
    shadowColor: "#5C8FB8",
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 8,
    elevation: 8,
    transform: [{ translateY: -2 }],
  },
  errorText: {
    color: "#B95C4A",
    fontSize: 14,
    marginBottom: 10,
  },
  downloadButtonPressed: {
    opacity: 0.8,
  },
});
