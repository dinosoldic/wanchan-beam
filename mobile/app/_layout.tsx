import { Stack, useRouter } from "expo-router";
import { Pressable, StyleSheet, View } from "react-native";

export default function RootLayout() {
  const router = useRouter();

  // styles
  const styles = StyleSheet.create({
    backButton: {
      marginLeft: 10,
      width: 40,
      height: 40,
      alignItems: "center",
      justifyContent: "center",
      backgroundColor: "none",
    },
    backButtonPressed: {
      backgroundColor: "none",
      transform: [{ scale: 1.2 }, { translateX: -10 }],
    },
    backArrow: {
      width: 15,
      height: 15,
      marginLeft: 4,
      borderLeftWidth: 2.5,
      borderBottomWidth: 2.5,
      borderColor: "#062653",
      transform: [{ rotate: "45deg" }],
    },
  });

  return (
    <Stack>
      <Stack.Screen name="index" options={{ headerShown: false }} />
      <Stack.Screen
        name="camera"
        options={{
          title: "",
          headerStyle: {
            backgroundColor: "#FFFFFF",
          },
          headerLeft: () => (
            <Pressable
              onPress={() => router.dismissTo("/")}
              accessibilityRole="button"
              accessibilityLabel="Go back"
              style={({ pressed }) => [
                styles.backButton,
                pressed && styles.backButtonPressed,
              ]}
            >
              <View style={styles.backArrow} />
            </Pressable>
          ),
        }}
      />
    </Stack>
  );
}
