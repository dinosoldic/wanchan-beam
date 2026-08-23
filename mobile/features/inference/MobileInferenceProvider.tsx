import {
  createContext,
  type PropsWithChildren,
  useContext,
} from "react";
import type { TensorflowPlugin } from "react-native-fast-tflite";

import mobileBreedClassifierAsset from "@/generated-assets/models/mobile-breed-classifier.tflite";
import mobileDetectorAsset from "@/generated-assets/models/mobile-detector.tflite";

import { useBundledTensorflowModel } from "./useBundledTensorflowModel";

interface MobileInferenceContextValue {
  mobileDetector: TensorflowPlugin;
  mobileBreedClassifier: TensorflowPlugin;
}

const MobileInferenceContext = createContext<
  MobileInferenceContextValue | undefined
>(undefined);

export function MobileInferenceProvider({ children }: PropsWithChildren) {
  // Keeping both models above the router lets live and static inference share
  // one native model instance instead of loading another copy per screen.
  // An empty delegate list keeps the tested CPU baseline on Android and iOS.
  const mobileDetector = useBundledTensorflowModel(mobileDetectorAsset);
  const mobileBreedClassifier = useBundledTensorflowModel(
    mobileBreedClassifierAsset,
  );

  return (
    <MobileInferenceContext.Provider
      value={{ mobileDetector, mobileBreedClassifier }}
    >
      {children}
    </MobileInferenceContext.Provider>
  );
}

export function useMobileInferenceModels(): MobileInferenceContextValue {
  const models = useContext(MobileInferenceContext);

  if (models === undefined) {
    throw new Error(
      "useMobileInferenceModels must be used inside MobileInferenceProvider.",
    );
  }

  return models;
}
