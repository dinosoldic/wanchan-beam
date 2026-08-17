import * as FileSystem from "expo-file-system/legacy";

export interface CapturedPhoto {
  uri: string;
  width: number;
  height: number;
}

let capturedPhoto: CapturedPhoto | null = null;

export function setCapturedPhoto(photo: CapturedPhoto): void {
  capturedPhoto = photo;
}

export function getCapturedPhoto(): CapturedPhoto | null {
  return capturedPhoto;
}

export async function discardCapturedPhoto(): Promise<void> {
  const photoToDelete = capturedPhoto;

  capturedPhoto = null;

  if (!photoToDelete) {
    return;
  }

  await FileSystem.deleteAsync(photoToDelete.uri, {
    idempotent: true,
  });
}
