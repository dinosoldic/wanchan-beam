import { useCallback, type RefObject } from "react";
import { isRunningInExpoGo } from "expo";
import * as Sharing from "expo-sharing";
import { captureRef } from "react-native-view-shot";
import { Alert, type View } from "react-native";

interface UseResultExportOptions {
  resultRef: RefObject<View | null>;
  setError: (message: string | null) => void;
}

export function useResultExport({
  resultRef,
  setError,
}: UseResultExportOptions) {
  const downloadResult = useCallback(async () => {
    if (!resultRef.current) {
      return;
    }

    setError(null);

    try {
      if (isRunningInExpoGo()) {
        setError("Photo access is required to save the scanned image.");
        return;
      }

      const localUri = await captureRef(resultRef, {
        format: "png",
        quality: 1,
        result: "tmpfile",
      });

      const MediaLibrary = await import("expo-media-library/legacy");

      const permission = await MediaLibrary.requestPermissionsAsync(true, [
        "photo",
      ]);

      if (!permission.granted) {
        setError("Photo access is required to save the scanned image.");
        return;
      }

      await MediaLibrary.saveToLibraryAsync(localUri);

      Alert.alert(
        "Image saved",
        "The scanned image was saved to your photo library.",
      );
    } catch (error) {
      console.warn("Download failed:", error);
      setError("Could not save the scanned image.");
    }
  }, [resultRef, setError]);

  const shareResult = useCallback(async () => {
    if (!resultRef.current) {
      return;
    }

    setError(null);

    try {
      const sharingAvailable = await Sharing.isAvailableAsync();

      if (!sharingAvailable) {
        setError("Sharing is not available on this device.");
        return;
      }

      const localUri = await captureRef(resultRef, {
        format: "png",
        quality: 1,
        result: "tmpfile",
      });

      await Sharing.shareAsync(localUri, {
        mimeType: "image/png",
        dialogTitle: "Share scanned image",
        UTI: "public.png",
      });
    } catch (error) {
      console.warn("Sharing failed:", error);
      setError("Could not share the scanned image.");
    }
  }, [resultRef, setError]);

  return {
    downloadResult,
    shareResult,
  };
}
