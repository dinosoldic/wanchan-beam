import { Ionicons } from "@expo/vector-icons";
import { Pressable, StyleSheet, Text, View } from "react-native";

interface RetryOverlayProps {
  onRetry: () => void;
}

export function RetryOverlay({ onRetry }: RetryOverlayProps) {
  return (
    <View style={styles.overlay}>
      <Pressable
        onPress={(event) => {
          event.stopPropagation();
          onRetry();
        }}
        accessibilityRole="button"
        accessibilityLabel="Retry dog detection"
        style={({ pressed }) => [
          styles.button,
          pressed && styles.buttonPressed,
        ]}
      >
        <Ionicons name="refresh-outline" size={23} color="#FFF8EE" />
        <Text style={styles.buttonText}>Retry</Text>
      </Pressable>
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
    zIndex: 10,
    elevation: 10,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "rgba(6, 38, 83, 0.28)",
  },
  button: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    borderRadius: 14,
    backgroundColor: "#F3A58F",
    paddingHorizontal: 22,
    paddingVertical: 13,
  },
  buttonPressed: {
    opacity: 0.8,
    transform: [{ scale: 0.96 }],
  },
  buttonText: {
    color: "#FFF8EE",
    fontSize: 16,
    fontWeight: "700",
  },
});
