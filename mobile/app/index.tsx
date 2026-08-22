import { useRouter } from "expo-router";
import { Image, Pressable, StyleSheet, Text } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

export default function HomeScreen() {
  const router = useRouter();

  return (
    <SafeAreaView style={styles.container}>
      <Image
        source={require("../assets/logo/splash-icon.png")}
        style={styles.logo}
        resizeMode="contain"
        accessibilityLabel="WanChan Beam logo"
      />

      <Pressable
        onPress={() => {
          router.push("/camera");
        }}
        accessibilityRole="button"
        accessibilityLabel="Start the camera"
        style={({ pressed }) => [
          styles.actionButton,
          pressed && styles.actionButtonPressed,
        ]}
      >
        <Text style={styles.actionButtonText}>Live Scan</Text>
      </Pressable>
      <Pressable
        onPress={() => {
          router.push("/upload");
        }}
        accessibilityRole="button"
        accessibilityLabel="Upload Image"
        style={({ pressed }) => [
          styles.actionButton,
          pressed && styles.actionButtonPressed,
        ]}
      >
        <Text style={styles.actionButtonText}>Upload Photo</Text>
      </Pressable>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#FFF8EE",
    padding: 24,
  },
  logo: {
    width: "70%",
    maxWidth: 300,
    maxHeight: 300,
    marginBottom: 20,
  },
  actionButton: {
    width: "60%",
    maxWidth: 300,
    alignItems: "center",
    borderRadius: 10,
    backgroundColor: "#F3A58F",
    paddingHorizontal: 40,
    paddingVertical: 16,
    marginBottom: 16,
  },
  actionButtonPressed: {
    opacity: 0.8,
    backgroundColor: "#E89079",
  },
  actionButtonText: {
    color: "#FFF8EE",
    fontSize: 18,
    fontWeight: "700",
  },
});
