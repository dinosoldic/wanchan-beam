import { Asset } from "expo-asset";
import { useEffect, useState } from "react";
import {
  loadTensorflowModel,
  type TensorflowPlugin,
} from "react-native-fast-tflite";

export function useBundledTensorflowModel(
  moduleId: number,
): TensorflowPlugin {
  const [plugin, setPlugin] = useState<TensorflowPlugin>({
    model: undefined,
    state: "loading",
  });

  useEffect(() => {
    let cancelled = false;

    async function loadModel(): Promise<void> {
      try {
        setPlugin({ model: undefined, state: "loading" });

        // Expo copies the embedded Android resource to a file:// cache path.
        const asset = Asset.fromModule(moduleId);
        await asset.downloadAsync();

        if (asset.localUri === null) {
          throw new Error("Bundled TFLite model has no local file URI.");
        }

        // Keep the tested CPU path instead of enabling a platform delegate.
        const model = await loadTensorflowModel({ url: asset.localUri }, []);

        if (!cancelled) {
          setPlugin({ model, state: "loaded" });
        }
      } catch (error) {
        if (!cancelled) {
          setPlugin({
            model: undefined,
            state: "error",
            error:
              error instanceof Error ? error : new Error(String(error)),
          });
        }
      }
    }

    void loadModel();

    return () => {
      cancelled = true;
    };
  }, [moduleId]);

  return plugin;
}
