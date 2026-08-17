import { Fragment } from "react";
import { StyleSheet, Text, View } from "react-native";

import type { DogDetectionResponse } from "@/types/detection";

interface DetectionOverlayProps {
  result: DogDetectionResponse;
  displayWidth: number;
  displayHeight: number;
}

const LABEL_WIDTH = 82;
const LABEL_HEIGHT = 26;

export function DetectionOverlay({
  result,
  displayWidth,
  displayHeight,
}: DetectionOverlayProps) {
  const scaleX = displayWidth / result.image.width;
  const scaleY = displayHeight / result.image.height;

  return (
    <View style={styles.overlay} pointerEvents="none">
      {result.detections.map((detection, index) => {
        const { x1, y1, x2, y2 } = detection.box;
        const boxLeft = x1 * scaleX;
        const boxTop = y1 * scaleY;
        const boxWidth = Math.max((x2 - x1) * scaleX, 1);
        const boxHeight = Math.max((y2 - y1) * scaleY, 1);
        const confidence = (detection.confidence * 100).toFixed(1);

        const labelLeft = Math.min(
          Math.max(boxLeft, 0),
          Math.max(displayWidth - LABEL_WIDTH, 0),
        );

        const labelTop =
          boxTop >= LABEL_HEIGHT
            ? boxTop - LABEL_HEIGHT
            : Math.min(boxTop + boxHeight, displayHeight - LABEL_HEIGHT);

        return (
          <Fragment key={`${index}-${x1}-${y1}`}>
            <View
              style={[
                styles.box,
                {
                  left: boxLeft,
                  top: boxTop,
                  width: boxWidth,
                  height: boxHeight,
                },
              ]}
            />

            <View
              style={[
                styles.label,
                {
                  left: labelLeft,
                  top: labelTop,
                },
              ]}
            >
              <Text numberOfLines={1} style={styles.labelText}>
                Dog {confidence}%
              </Text>
            </View>
          </Fragment>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    overflow: "hidden",
  },
  box: {
    position: "absolute",
    borderWidth: 3,
    borderColor: "#F3A58F",
    borderRadius: 8,
    backgroundColor: "rgba(243, 165, 143, 0.08)",
  },
  label: {
    position: "absolute",
    width: LABEL_WIDTH,
    height: LABEL_HEIGHT,
    alignItems: "center",
    justifyContent: "center",
    borderRadius: 7,
    backgroundColor: "#F3A58F",
  },
  labelText: {
    color: "#FFF8EE",
    fontSize: 12,
    fontWeight: "800",
  },
});
