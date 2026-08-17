import { useRef, useState } from "react";
import { useCameraPermissions, CameraView } from "expo-camera";
import { Image, Pressable, StyleSheet, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { detectDogs } from "@/services/RemoteInferenceService";

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isCameraReady, setIsCameraReady] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanStatus, setScanStatus] = useState("Camera starting...");

  async function handleScan() {
    if (!cameraRef.current || !isCameraReady || isScanning) {
      return;
    }

    setIsScanning(true);
    setScanStatus("Scanning...");

    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });

      const result = await detectDogs(photo.uri);
      const dogCount = result.detections.length;

      console.log("Detection result:", result);

      setScanStatus(`${dogCount} ${dogCount === 1 ? "dog" : "dogs"} detected`);
    } catch (error) {
      console.error("Detection failed:", error);
      setScanStatus("Scan failed");
    } finally {
      setIsScanning(false);
    }
  }

  if (!permission) {
    return <SafeAreaView style={styles.container} />;
  }

  if (!permission.granted) {
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
            onPress={requestPermission}
            style={styles.permissionButton}
          >
            <Text style={styles.permissionButtonText}>Allow camera</Text>
          </Pressable>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.cameraView}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing="back"
          onCameraReady={() => {
            setIsCameraReady(true);
            setScanStatus("Ready to scan");
          }}
        />
        <View style={[styles.corner, styles.topLeft]} />
        <View style={[styles.corner, styles.topRight]} />
        <View style={[styles.corner, styles.bottomLeft]} />
        <View style={[styles.corner, styles.bottomRight]} />
        <View style={styles.scanControls}>
          <Text style={styles.scanStatus}>{scanStatus}</Text>

          <Pressable
            accessibilityRole="button"
            accessibilityLabel="Scan for dogs"
            disabled={!isCameraReady || isScanning}
            onPress={handleScan}
            style={({ pressed }) => [
              styles.scanButton,
              (!isCameraReady || isScanning) && styles.scanButtonDisabled,
              pressed && styles.scanButtonPressed,
            ]}
          >
            <Text style={styles.scanButtonText}>
              {isScanning ? "Scanning..." : "Scan"}
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
  scanStatus: {
    color: "#FFF8EE",
    fontSize: 14,
    fontWeight: "600",
    backgroundColor: "rgba(6, 38, 83, 0.75)",
    paddingHorizontal: 14,
    paddingVertical: 7,
    borderRadius: 999,
  },
  scanButton: {
    minWidth: 120,
    alignItems: "center",
    backgroundColor: "#F3A58F",
    borderRadius: 999,
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
    color: "#062653",
    fontSize: 16,
    fontWeight: "700",
  },
});
