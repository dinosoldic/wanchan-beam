import {
  loadImage,
  type Image,
} from "react-native-nitro-image";

const DETECTOR_INPUT_SIZE = 544;
const DETECTOR_CHANNELS = 3;
const DETECTOR_PADDING_VALUE = 114 / 255;

interface PixelChannelLayout {
  bytesPerPixel: number;
  redOffset: number;
  greenOffset: number;
  blueOffset: number;
}

type PixelFormat = ReturnType<Image["toRawPixelData"]>["pixelFormat"];

export interface StaticDetectorTransform {
  originalWidth: number;
  originalHeight: number;
  resizedWidth: number;
  resizedHeight: number;
  paddingLeft: number;
  paddingTop: number;
}

export interface StaticDetectorInput {
  buffer: ArrayBuffer;
  transform: StaticDetectorTransform;
}

function getPixelChannelLayout(
  pixelFormat: PixelFormat,
): PixelChannelLayout {
  switch (pixelFormat) {
    case "RGB":
      return {
        bytesPerPixel: 3,
        redOffset: 0,
        greenOffset: 1,
        blueOffset: 2,
      };
    case "BGR":
      return {
        bytesPerPixel: 3,
        redOffset: 2,
        greenOffset: 1,
        blueOffset: 0,
      };
    case "RGBA":
    case "RGBX":
      return {
        bytesPerPixel: 4,
        redOffset: 0,
        greenOffset: 1,
        blueOffset: 2,
      };
    case "BGRA":
    case "BGRX":
      return {
        bytesPerPixel: 4,
        redOffset: 2,
        greenOffset: 1,
        blueOffset: 0,
      };
    case "ARGB":
    case "XRGB":
      return {
        bytesPerPixel: 4,
        redOffset: 1,
        greenOffset: 2,
        blueOffset: 3,
      };
    case "ABGR":
    case "XBGR":
      return {
        bytesPerPixel: 4,
        redOffset: 3,
        greenOffset: 2,
        blueOffset: 1,
      };
    default:
      throw new Error(`Unsupported static image pixel format: ${pixelFormat}`);
  }
}

export async function createStaticDetectorInput(
  imageUri: string,
): Promise<StaticDetectorInput> {
  const sourceImage = await loadImage({ filePath: imageUri });
  let resizedImage: Image | undefined;

  try {
    const originalWidth = sourceImage.width;
    const originalHeight = sourceImage.height;

    if (originalWidth <= 0 || originalHeight <= 0) {
      throw new Error(
        `Invalid static image size: ${originalWidth}x${originalHeight}.`,
      );
    }

    const scale = Math.min(
      DETECTOR_INPUT_SIZE / originalWidth,
      DETECTOR_INPUT_SIZE / originalHeight,
    );

    const resizedWidth = Math.max(1, Math.round(originalWidth * scale));
    const resizedHeight = Math.max(1, Math.round(originalHeight * scale));
    const paddingLeft = Math.floor(
      (DETECTOR_INPUT_SIZE - resizedWidth) / 2,
    );
    const paddingTop = Math.floor(
      (DETECTOR_INPUT_SIZE - resizedHeight) / 2,
    );

    resizedImage = await sourceImage.resizeAsync(resizedWidth, resizedHeight);

    const rawPixels = await resizedImage.toRawPixelDataAsync();
    const channelLayout = getPixelChannelLayout(rawPixels.pixelFormat);
    const sourceValues = new Uint8Array(rawPixels.buffer);
    const expectedSourceBytes =
      resizedWidth * resizedHeight * channelLayout.bytesPerPixel;

    if (
      rawPixels.width !== resizedWidth ||
      rawPixels.height !== resizedHeight ||
      sourceValues.byteLength !== expectedSourceBytes
    ) {
      throw new Error(
        `Unexpected resized image buffer: ${rawPixels.width}x${rawPixels.height}, ` +
          `${sourceValues.byteLength} bytes.`,
      );
    }

    const planeSize = DETECTOR_INPUT_SIZE * DETECTOR_INPUT_SIZE;
    const detectorValues = new Float32Array(planeSize * DETECTOR_CHANNELS);

    detectorValues.fill(DETECTOR_PADDING_VALUE);

    for (let y = 0; y < resizedHeight; y += 1) {
      for (let x = 0; x < resizedWidth; x += 1) {
        const sourceIndex =
          (y * resizedWidth + x) * channelLayout.bytesPerPixel;
        const outputIndex =
          (paddingTop + y) * DETECTOR_INPUT_SIZE + paddingLeft + x;

        detectorValues[outputIndex] =
          sourceValues[sourceIndex + channelLayout.redOffset]! / 255;
        detectorValues[planeSize + outputIndex] =
          sourceValues[sourceIndex + channelLayout.greenOffset]! / 255;
        detectorValues[planeSize * 2 + outputIndex] =
          sourceValues[sourceIndex + channelLayout.blueOffset]! / 255;
      }
    }

    return {
      buffer: detectorValues.buffer,
      transform: {
        originalWidth,
        originalHeight,
        resizedWidth,
        resizedHeight,
        paddingLeft,
        paddingTop,
      },
    };
  } finally {
    if (resizedImage !== undefined && resizedImage !== sourceImage) {
      resizedImage.dispose();
    }

    sourceImage.dispose();
  }
}
