import { fetch } from "expo/fetch";
import { File } from "expo-file-system";

import type { DogDetectionResponse } from "@/types/detection";

const DETECTION_TIMEOUT_MS = 5000;

function getApiUrl(): string {
  const apiUrl = process.env.EXPO_PUBLIC_API_URL;

  if (!apiUrl) {
    throw new Error("EXPO_PUBLIC_API_URL is not configured");
  }

  return apiUrl.replace(/\/+$/, "");
}

function getImageFile(imageUri: string): File {
  return new File(imageUri);
}

export async function detectDogs(
  imageUri: string,
): Promise<DogDetectionResponse> {
  const formData = new FormData();
  const imageFile = getImageFile(imageUri);

  formData.append("image", imageFile, "camera-frame.jpg");

  // DetectionService turns this timeout into an on-device fallback.
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => {
    abortController.abort();
  }, DETECTION_TIMEOUT_MS);

  try {
    const response = await fetch(`${getApiUrl()}/detect`, {
      method: "POST",
      body: formData,
      signal: abortController.signal,
    });

    if (!response.ok) {
      const details = await response.text();

      throw new Error(
        `Detection request failed (${response.status}): ${details}`,
      );
    }

    return (await response.json()) as DogDetectionResponse;
  } catch (error) {
    if (abortController.signal.aborted) {
      throw new Error("Detection request timed out.");
    }

    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}
