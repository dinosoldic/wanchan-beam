import { useEffect, useState } from "react";
import { Animated, Easing, StyleSheet, View } from "react-native";
import { LinearGradient } from "expo-linear-gradient";

const SCAN_TRAVEL_DURATION_MS = 900;
const SCAN_LINE_HEIGHT = 80;

interface ScanningOverlayProps {
  imageHeight: number;
}

export function ScanningOverlay({ imageHeight }: ScanningOverlayProps) {
  const [scanProgress] = useState(() => new Animated.Value(0));

  useEffect(() => {
    const animation = Animated.loop(
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

    animation.start();

    return () => {
      animation.stop();
      scanProgress.setValue(0);
    };
  }, [scanProgress]);

  const scanLinePosition = scanProgress.interpolate({
    inputRange: [0, 1],
    outputRange: [0, Math.max(imageHeight - SCAN_LINE_HEIGHT, 0)],
  });

  return (
    <View style={styles.overlay} pointerEvents="none">
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
  );
}

const styles = StyleSheet.create({
  overlay: {
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
});
