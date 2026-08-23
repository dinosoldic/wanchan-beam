import type { TfliteModel } from "react-native-fast-tflite";

import { detectDogsLocally } from "@/features/inference";
import type { DogDetectionResponse } from "@/types/detection";

import { detectDogs as detectDogsRemotely } from "./RemoteInferenceService";

export interface DetectionModels {
  detectorModel: TfliteModel | undefined;
  breedClassifierModel: TfliteModel | undefined;
}

export async function detectDogs(
  imageUri: string,
  models: DetectionModels,
): Promise<DogDetectionResponse> {
  try {
    return await detectDogsRemotely(imageUri);
  } catch (remoteError) {
    if (
      models.detectorModel === undefined ||
      models.breedClassifierModel === undefined
    ) {
      const remoteMessage =
        remoteError instanceof Error ? remoteError.message : String(remoteError);

      throw new Error(
        `Remote detection failed and mobile models are not ready: ${remoteMessage}`,
      );
    }

    return detectDogsLocally(imageUri, {
      detectorModel: models.detectorModel,
      breedClassifierModel: models.breedClassifierModel,
    });
  }
}
