import { StyleSheet, Text, View, type ViewStyle } from "react-native";

import type { DogDetectionResponse } from "@/types/detection";

interface DetectionOverlayProps {
  result: DogDetectionResponse;
  displayWidth: number;
  displayHeight: number;
}

type CalloutSide = "left" | "right";

interface DetectionAnchor {
  key: string;
  anchorX: number;
  anchorY: number;
  breedLabel: string;
}

interface DetectionCallout extends DetectionAnchor {
  side: CalloutSide;
  labelLeft: number;
  labelTop: number;
  labelWidth: number;
  lineEndX: number;
  lineEndY: number;
}

const LABEL_MAX_WIDTH = 132;
const LABEL_WIDTH_RATIO = 0.36;
const LABEL_SLOT_HEIGHT = 42;
const EDGE_PADDING = 6;
const DOT_SIZE = 9;
const LINE_THICKNESS = 2;

function formatBreedLabel(label: string): string {
  return label
    .split("_")
    .map(
      (word) => `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`,
    )
    .join(" ");
}

function positionCalloutSide(
  anchors: readonly DetectionAnchor[],
  side: CalloutSide,
  displayWidth: number,
  displayHeight: number,
  labelWidth: number,
): DetectionCallout[] {
  const sortedAnchors = [...anchors].sort(
    (first, second) => first.anchorY - second.anchorY,
  );

  const availableTravel = Math.max(
    displayHeight - EDGE_PADDING * 2 - LABEL_SLOT_HEIGHT,
    0,
  );

  const labelLeft =
    side === "left"
      ? EDGE_PADDING
      : Math.max(displayWidth - EDGE_PADDING - labelWidth, 0);

  return sortedAnchors.map((anchor, index) => {
    const labelTop =
      sortedAnchors.length === 1
        ? EDGE_PADDING + availableTravel / 2
        : EDGE_PADDING + (availableTravel * index) / (sortedAnchors.length - 1);

    return {
      ...anchor,
      side,
      labelLeft,
      labelTop,
      labelWidth,
      lineEndX: side === "left" ? labelLeft + labelWidth : labelLeft,
      lineEndY: labelTop + LABEL_SLOT_HEIGHT / 2,
    };
  });
}

function buildDetectionCallouts(
  result: DogDetectionResponse,
  displayWidth: number,
  displayHeight: number,
): DetectionCallout[] {
  const scaleX = displayWidth / result.image.width;
  const scaleY = displayHeight / result.image.height;
  const labelWidth = Math.min(
    LABEL_MAX_WIDTH,
    Math.max(displayWidth * LABEL_WIDTH_RATIO, 1),
  );

  const anchors = result.detections
    .map((detection, index): DetectionAnchor => {
      const { x1, y1, x2, y2 } = detection.box;
      const [firstBreed] = detection.breedPredictions;

      return {
        key: `${index}-${x1}-${y1}`,
        anchorX: ((x1 + x2) / 2) * scaleX,
        anchorY: ((y1 + y2) / 2) * scaleY,
        breedLabel: formatBreedLabel(firstBreed.label),
      };
    })
    .sort((first, second) => first.anchorX - second.anchorX);

  // Keep the two sides balanced while assigning leftmost dogs to the left
  // and rightmost dogs to the right, which minimizes leader-line crossings.
  const middleIndex = Math.floor(anchors.length / 2);
  const middleAnchor = anchors[middleIndex];
  const leftGetsExtra =
    anchors.length % 2 === 1 &&
    middleAnchor !== undefined &&
    middleAnchor.anchorX <= displayWidth / 2;
  const leftCount = middleIndex + (leftGetsExtra ? 1 : 0);

  const leftCallouts = positionCalloutSide(
    anchors.slice(0, leftCount),
    "left",
    displayWidth,
    displayHeight,
    labelWidth,
  );

  const rightCallouts = positionCalloutSide(
    anchors.slice(leftCount),
    "right",
    displayWidth,
    displayHeight,
    labelWidth,
  );

  return [...leftCallouts, ...rightCallouts];
}

function createLineStyle(callout: DetectionCallout): ViewStyle {
  const deltaX = callout.lineEndX - callout.anchorX;
  const deltaY = callout.lineEndY - callout.anchorY;
  const length = Math.hypot(deltaX, deltaY);
  const midpointX = (callout.anchorX + callout.lineEndX) / 2;
  const midpointY = (callout.anchorY + callout.lineEndY) / 2;
  const angle = (Math.atan2(deltaY, deltaX) * 180) / Math.PI;

  return {
    left: midpointX - length / 2,
    top: midpointY - LINE_THICKNESS / 2,
    width: length,
    transform: [{ rotate: `${angle}deg` }],
  };
}

export function DetectionOverlay({
  result,
  displayWidth,
  displayHeight,
}: DetectionOverlayProps) {
  const callouts = buildDetectionCallouts(result, displayWidth, displayHeight);

  return (
    <View style={styles.overlay} pointerEvents="none">
      {callouts.map((callout) => (
        <View
          key={`${callout.key}-line`}
          style={[styles.line, createLineStyle(callout)]}
        />
      ))}

      {callouts.map((callout) => (
        <View
          key={`${callout.key}-dot`}
          style={[
            styles.dot,
            {
              left: callout.anchorX - DOT_SIZE / 2,
              top: callout.anchorY - DOT_SIZE / 2,
            },
          ]}
        />
      ))}

      {callouts.map((callout) => (
        <View
          key={`${callout.key}-label`}
          style={[
            styles.labelSlot,
            {
              left: callout.labelLeft,
              top: callout.labelTop,
              width: callout.labelWidth,
            },
          ]}
        >
          <View style={styles.label}>
            <Text
              numberOfLines={2}
              ellipsizeMode="tail"
              style={styles.labelText}
            >
              {callout.breedLabel}
            </Text>
          </View>
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    overflow: "hidden",
  },
  line: {
    position: "absolute",
    height: LINE_THICKNESS,
    backgroundColor: "#F3A58F",
  },
  dot: {
    position: "absolute",
    width: DOT_SIZE,
    height: DOT_SIZE,
    borderWidth: 2,
    borderColor: "#FFF8EE",
    borderRadius: DOT_SIZE / 2,
    backgroundColor: "#F3A58F",
  },
  labelSlot: {
    position: "absolute",
    height: LABEL_SLOT_HEIGHT,
    alignItems: "center",
    justifyContent: "center",
  },
  label: {
    width: "100%",
    paddingVertical: 6,
    borderRadius: 7,
    backgroundColor: "#F3A58F",
  },
  labelText: {
    width: "100%",
    paddingHorizontal: 6,
    color: "#FFF8EE",
    fontSize: 11,
    lineHeight: 14,
    fontWeight: "800",
    textAlign: "center",
  },
});
