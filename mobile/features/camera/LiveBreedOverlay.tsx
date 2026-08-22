import { useEffect, useState } from "react";
import { StyleSheet, Text, View, type ViewStyle } from "react-native";

import type {
  LiveBreedPrediction,
  LiveDetectionBox,
} from "@/features/inference";

interface LiveBreedOverlayPrediction extends LiveBreedPrediction {
  label: string;
}

export interface LiveBreedOverlayDetection {
  trackId?: number;
  box: LiveDetectionBox;
  breedPrediction: LiveBreedOverlayPrediction | null;
}

interface LiveBreedOverlayProps {
  detections: readonly LiveBreedOverlayDetection[];
  previewWidth: number;
  previewHeight: number;
}

type CalloutSide = "left" | "right";

interface LiveBreedAnchor {
  key: string;
  anchorX: number;
  anchorY: number;
  breedLabel: string | null;
}

interface LiveBreedCallout extends LiveBreedAnchor {
  labelLeft: number;
  labelTop: number;
  labelWidth: number;
  lineEndX: number;
  lineEndY: number;
}

/// consts
const LABEL_WIDTH = 132;
const LABEL_SLOT_HEIGHT = 42;
const EDGE_PADDING = 6;
const MINIMUM_BREED_CONFIDENCE = 0.4;

const CROWDED_ENTER_COUNT = 4;
const CROWDED_EXIT_COUNT = 2;
const CALLOUT_TOP_PADDING = 64;
const CALLOUT_BOTTOM_PADDING = 12;
const DOT_SIZE = 9;
const LINE_THICKNESS = 2;

/// funcs
function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

function formatBreedLabel(prediction: LiveBreedOverlayPrediction): string {
  if (prediction.confidence < MINIMUM_BREED_CONFIDENCE) {
    return "Breed uncertain";
  }

  return prediction.label
    .split("_")
    .map(
      (word) => `${word.charAt(0).toUpperCase()}${word.slice(1).toLowerCase()}`,
    )
    .join(" ");
}

function getDirectLabelPosition(
  box: LiveDetectionBox,
  previewWidth: number,
  previewHeight: number,
) {
  const centerX = (box.x1 + box.x2) / 2;
  const centerY = (box.y1 + box.y2) / 2;
  const dogWidth = Math.max(box.x2 - box.x1, 1);

  // Keep the pill compact for smaller dogs while leaving enough room for
  // breed names to wrap onto two lines.
  const labelWidth = Math.min(
    LABEL_WIDTH,
    Math.max(dogWidth, 88),
    Math.max(previewWidth - EDGE_PADDING * 2, 1),
  );

  return {
    left: clamp(
      centerX - labelWidth / 2,
      EDGE_PADDING,
      Math.max(previewWidth - labelWidth - EDGE_PADDING, EDGE_PADDING),
    ),
    top: clamp(
      centerY - LABEL_SLOT_HEIGHT / 2,
      EDGE_PADDING,
      Math.max(previewHeight - LABEL_SLOT_HEIGHT - EDGE_PADDING, EDGE_PADDING),
    ),
    width: labelWidth,
  };
}

function positionCalloutSide(
  anchors: readonly LiveBreedAnchor[],
  side: CalloutSide,
  previewWidth: number,
  previewHeight: number,
  labelWidth: number,
): LiveBreedCallout[] {
  const sortedAnchors = [...anchors].sort(
    (first, second) => first.anchorY - second.anchorY,
  );

  const availableTravel = Math.max(
    previewHeight -
      CALLOUT_TOP_PADDING -
      CALLOUT_BOTTOM_PADDING -
      LABEL_SLOT_HEIGHT,
    0,
  );

  const labelLeft =
    side === "left"
      ? EDGE_PADDING
      : Math.max(previewWidth - EDGE_PADDING - labelWidth, 0);

  return sortedAnchors.map((anchor, index) => {
    const labelTop =
      sortedAnchors.length === 1
        ? CALLOUT_TOP_PADDING + availableTravel / 2
        : CALLOUT_TOP_PADDING +
          (availableTravel * index) / (sortedAnchors.length - 1);

    return {
      ...anchor,
      labelLeft,
      labelTop,
      labelWidth,
      lineEndX: side === "left" ? labelLeft + labelWidth : labelLeft,
      lineEndY: labelTop + LABEL_SLOT_HEIGHT / 2,
    };
  });
}

function buildCrowdedCallouts(
  detections: readonly LiveBreedOverlayDetection[],
  previewWidth: number,
  previewHeight: number,
): LiveBreedCallout[] {
  const labelWidth = Math.min(
    LABEL_WIDTH,
    Math.max(previewWidth - EDGE_PADDING * 2, 1),
  );

  const anchors = detections
    .map(
      (detection, index): LiveBreedAnchor => ({
        key: `live-breed-${detection.trackId ?? index}`,
        anchorX: (detection.box.x1 + detection.box.x2) / 2,
        anchorY: (detection.box.y1 + detection.box.y2) / 2,
        breedLabel:
          detection.breedPrediction === null
            ? null
            : formatBreedLabel(detection.breedPrediction),
      }),
    )
    .sort((first, second) => first.anchorX - second.anchorX);

  const middleIndex = Math.floor(anchors.length / 2);
  const middleAnchor = anchors[middleIndex];

  const leftGetsExtra =
    anchors.length % 2 === 1 &&
    middleAnchor !== undefined &&
    middleAnchor.anchorX <= previewWidth / 2;

  const leftCount = middleIndex + (leftGetsExtra ? 1 : 0);

  return [
    ...positionCalloutSide(
      anchors.slice(0, leftCount),
      "left",
      previewWidth,
      previewHeight,
      labelWidth,
    ),
    ...positionCalloutSide(
      anchors.slice(leftCount),
      "right",
      previewWidth,
      previewHeight,
      labelWidth,
    ),
  ];
}

function createCalloutLineStyle(callout: LiveBreedCallout): ViewStyle {
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

export function LiveBreedOverlay({
  detections,
  previewWidth,
  previewHeight,
}: LiveBreedOverlayProps) {
  const [useCrowdedLayout, setUseCrowdedLayout] = useState(false);

  // Hysteresis prevents rapid layout switching when the count alternates
  // between three and four detections.
  useEffect(() => {
    setUseCrowdedLayout((currentLayout) => {
      if (detections.length >= CROWDED_ENTER_COUNT) {
        return true;
      }

      if (detections.length <= CROWDED_EXIT_COUNT) {
        return false;
      }

      return currentLayout;
    });
  }, [detections.length]);

  const crowdedCallouts = useCrowdedLayout
    ? buildCrowdedCallouts(detections, previewWidth, previewHeight)
    : [];

  const visibleCrowdedCallouts = crowdedCallouts.filter(
    (callout) => callout.breedLabel !== null,
  );

  return (
    <View pointerEvents="none" style={styles.overlay}>
      {useCrowdedLayout ? (
        <>
          {visibleCrowdedCallouts.map((callout) => (
            <View
              key={`${callout.key}-line`}
              style={[styles.line, createCalloutLineStyle(callout)]}
            />
          ))}

          {visibleCrowdedCallouts.map((callout) => (
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

          {visibleCrowdedCallouts.map((callout) => (
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
        </>
      ) : (
        detections.map((detection, index) => {
          if (detection.breedPrediction === null) {
            return null;
          }

          const position = getDirectLabelPosition(
            detection.box,
            previewWidth,
            previewHeight,
          );

          return (
            <View
              key={`live-breed-${detection.trackId ?? index}`}
              style={[styles.labelSlot, position]}
            >
              <View style={styles.label}>
                <Text
                  numberOfLines={2}
                  ellipsizeMode="tail"
                  style={styles.labelText}
                >
                  {formatBreedLabel(detection.breedPrediction)}
                </Text>
              </View>
            </View>
          );
        })
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  overlay: {
    ...StyleSheet.absoluteFill,
    overflow: "hidden",
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
});
