import { useCallback, useMemo, useState } from "react";
import * as FileSystem from "expo-file-system/legacy";
import { useFocusEffect, useRouter } from "expo-router";
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
} from "react-native-vision-camera";
import { SafeAreaView } from "react-native-safe-area-context";

import { setCapturedPhoto } from "@/features/camera/capturedPhotoStore";

export default function CameraScreen() {
  const router = useRouter();
  const { hasPermission, canRequestPermission, requestPermission } =
    useCameraPermission();

  const photoOutput = usePhotoOutput({
    containerFormat: "jpeg",
    quality: 0.9,
    qualityPrioritization: "quality",
  });
  const cameraOutputs = useMemo(() => [photoOutput], [photoOutput]);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);

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
