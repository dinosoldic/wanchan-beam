import { useEffect, useState } from "react";
import * as ImagePicker from "expo-image-picker";
import { LinearGradient } from "expo-linear-gradient";
import {
  Animated,
  Easing,
  Image,
  Pressable,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { detectDogs } from "@/services/RemoteInferenceService";
import type { DogDetectionResponse } from "@/types/detection";

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

  const detectedDogCount = detectionResult?.detections.length ?? 0;
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
        {imageUri ? (
          <Image
            source={{ uri: imageUri }}
            style={styles.previewImage}
            resizeMode="contain"
            accessibilityLabel="Selected photo"
          />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderTitle}>Choose a photo</Text>
            <Text style={styles.placeholderMessage}>
              Tap here to select an image from your device
            </Text>
          </View>
        )}

        {detectionResult && (
          <View style={styles.resultBadge} pointerEvents="none">
            <Text style={styles.resultBadgeText}>
              {detectedDogCount} {detectedDogCount === 1 ? "dog" : "dogs"}{" "}
              detected
            </Text>
          </View>
        )}

        {isScanning && containedImageLayout && (
          <View
            style={[styles.scanningOverlay, containedImageLayout]}
            pointerEvents="none"
          >
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
              disabled
              accessibilityRole="button"
              accessibilityLabel="Download scanned image"
              accessibilityState={{ disabled: true }}
              style={[styles.downloadButton, styles.downloadButtonDisabled]}
            >
              <Text style={styles.downloadIcon}>{"\u2193"}</Text>
              <View style={styles.downloadLine} />
            </Pressable>
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
    width: "70%",
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
  downloadButtonDisabled: {
    opacity: 0.45,
  },
  downloadIcon: {
    color: "#FFF8EE",
    fontSize: 25,
    fontWeight: "700",
    lineHeight: 25,
  },
  downloadLine: {
    width: 18,
    height: 2,
    borderRadius: 999,
    backgroundColor: "#FFF8EE",
  },
  resultBadge: {
    position: "absolute",
    bottom: 16,
    alignSelf: "center",
    borderRadius: 999,
    backgroundColor: "rgba(6, 38, 83, 0.82)",
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  resultBadgeText: {
    color: "#FFF8EE",
    fontSize: 14,
    fontWeight: "700",
  },
  scanningOverlay: {
    position: "absolute",
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
});
