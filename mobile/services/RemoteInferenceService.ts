import { fetch } from "expo/fetch";
import { Platform } from "react-native";

import type { DogDetectionResponse } from "@/types/detection";

function getApiUrl(): string {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL;

  if (!apiUrl) {
    throw new Error("EXPO_PUBLIC_API_URL is not configured");
  }

  return apiUrl.replace(/\/+$/, "");
}

async function getImageFile(imageUri: string): Promise<Blob> {
  if (Platform.OS === "web") {
    const response = await fetch(imageUri);

    if (!response.ok) {
      throw new Error("Could not read the selected image");
    }

    return response.blob();
  }

  const { File } = await import("expo-file-system");

  return new File(imageUri);
}

export async function detectDogs(
  imageUri: string,
): Promise<DogDetectionResponse> {
  const formData = new FormData();
  const imageFile = await getImageFile(imageUri);

  formData.append("image", imageFile, "camera-frame.jpg");

  const response = await fetch(`${getApiUrl()}/detect`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const details = await response.text();

    throw new Error(
      `Detection request failed (${response.status}): ${details}`,
    );
  }

  return (await response.json()) as DogDetectionResponse;
}
