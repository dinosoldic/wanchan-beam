import { fetch } from "expo/fetch";
import { File } from "expo-file-system";

import type { DogDetectionResponse } from "@/types/detection";

function getApiUrl(): string {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL;

  if (!apiUrl) {
    throw new Error("EXPO_PUBLIC_API_URL is not configured");
  }

  return apiUrl.replace(/\/+$/, "");
}

export async function detectDogs(
  imageUri: string,
): Promise<DogDetectionResponse> {
  const formData = new FormData();
  const imageFile = new File(imageUri);

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
