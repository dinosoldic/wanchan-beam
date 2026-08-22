import sharp from "sharp";

const INPUT_SIZE = 256;
const CHANNEL_COUNT = 3;
const PADDING_RED = 124;
const PADDING_GREEN = 116;
const PADDING_BLUE = 104;

export interface BreedCropBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface BreedCropTransform {
  originalWidth: number;
  originalHeight: number;
  cropLeft: number;
  cropTop: number;
  cropWidth: number;
  cropHeight: number;
  resizedWidth: number;
  resizedHeight: number;
  paddingLeft: number;
  paddingTop: number;
  paddingRight: number;
  paddingBottom: number;
}

export interface LetterboxedBreedCrop {
  pixels: Buffer;
  transform: BreedCropTransform;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}

/**
 * JavaScript's Math.round uses a different tie-breaking rule from Python.
 * Python rounds exact half values to the nearest even integer.
 */
function roundHalfToEven(value: number): number {
  const lowerInteger = Math.floor(value);
  const fraction = value - lowerInteger;

  if (fraction < 0.5) {
    return lowerInteger;
  }

  if (fraction > 0.5) {
    return lowerInteger + 1;
  }

  return lowerInteger % 2 === 0 ? lowerInteger : lowerInteger + 1;
}

function validateBox(box: BreedCropBox): void {
  const coordinates = [box.x1, box.y1, box.x2, box.y2];

  if (!coordinates.every(Number.isFinite)) {
    throw new TypeError("Breed crop coordinates must be finite numbers");
  }

  if (box.x2 <= box.x1 || box.y2 <= box.y1) {
    throw new RangeError("Breed crop box must have positive width and height");
  }
}

export async function letterboxBreedCrop(
  input: Buffer,
  box: BreedCropBox,
): Promise<LetterboxedBreedCrop> {
  validateBox(box);

  const metadata = await sharp(input, {
    failOn: "error",
  }).metadata();

  // The detector also uses auto-oriented dimensions, so its coordinates
  // refer to this same image orientation.
  const originalWidth = metadata.autoOrient.width;
  const originalHeight = metadata.autoOrient.height;

  // Round outward to avoid trimming part of the detected dog.
  const cropLeft = clamp(Math.floor(box.x1), 0, originalWidth);
  const cropTop = clamp(Math.floor(box.y1), 0, originalHeight);
  const cropRight = clamp(Math.ceil(box.x2), 0, originalWidth);
  const cropBottom = clamp(Math.ceil(box.y2), 0, originalHeight);

  const cropWidth = cropRight - cropLeft;
  const cropHeight = cropBottom - cropTop;

  if (cropWidth <= 0 || cropHeight <= 0) {
    throw new RangeError("Breed crop box has no area after clamping");
  }

  const scale = Math.min(INPUT_SIZE / cropWidth, INPUT_SIZE / cropHeight);

  // Match Python's round() behavior from the training loader.
  const resizedWidth = Math.max(1, roundHalfToEven(cropWidth * scale));
  const resizedHeight = Math.max(1, roundHalfToEven(cropHeight * scale));

  const horizontalPadding = INPUT_SIZE - resizedWidth;
  const verticalPadding = INPUT_SIZE - resizedHeight;

  const paddingLeft = Math.floor(horizontalPadding / 2);
  const paddingRight = horizontalPadding - paddingLeft;
  const paddingTop = Math.floor(verticalPadding / 2);
  const paddingBottom = verticalPadding - paddingTop;

  const background = {
    r: PADDING_RED,
    g: PADDING_GREEN,
    b: PADDING_BLUE,
  };

  const { data, info } = await sharp(input, {
    failOn: "error",
  })
    .autoOrient()
    .extract({
      left: cropLeft,
      top: cropTop,
      width: cropWidth,
      height: cropHeight,
    })
    .resize(resizedWidth, resizedHeight, {
      fit: "fill",
      kernel: sharp.kernel.linear,
    })
    .extend({
      top: paddingTop,
      bottom: paddingBottom,
      left: paddingLeft,
      right: paddingRight,
      background,
    })
    .flatten({
      background,
    })
    .toColourspace("srgb")
    .raw()
    .toBuffer({
      resolveWithObject: true,
    });

  if (
    info.width !== INPUT_SIZE ||
    info.height !== INPUT_SIZE ||
    info.channels !== CHANNEL_COUNT
  ) {
    throw new Error(
      "Unexpected breed crop output: " +
        `${info.width}x${info.height}x${info.channels}`,
    );
  }

  return {
    pixels: data,
    transform: {
      originalWidth,
      originalHeight,
      cropLeft,
      cropTop,
      cropWidth,
      cropHeight,
      resizedWidth,
      resizedHeight,
      paddingLeft,
      paddingTop,
      paddingRight,
      paddingBottom,
    },
  };
}
