import sharp from "sharp";

const INPUT_SIZE = 960;
const PADDING_VALUE = 114;

export interface LetterboxTransform {
  originalWidth: number;
  originalHeight: number;
  resizedWidth: number;
  resizedHeight: number;
  scale: number;
  paddingLeft: number;
  paddingTop: number;
  paddingRight: number;
  paddingBottom: number;
}

export interface LetterboxedImage {
  pixels: Buffer;
  transform: LetterboxTransform;
}

export async function letterboxImage(input: Buffer): Promise<LetterboxedImage> {
  const metadata = await sharp(input, {
    failOn: "error",
  }).metadata();

  const originalWidth = metadata.autoOrient.width;
  const originalHeight = metadata.autoOrient.height;

  const scale = Math.min(
    INPUT_SIZE / originalWidth,
    INPUT_SIZE / originalHeight,
  );

  const resizedWidth = Math.round(originalWidth * scale);
  const resizedHeight = Math.round(originalHeight * scale);

  const horizontalPadding = INPUT_SIZE - resizedWidth;
  const verticalPadding = INPUT_SIZE - resizedHeight;

  const paddingLeft = Math.floor(horizontalPadding / 2);
  const paddingRight = horizontalPadding - paddingLeft;
  const paddingTop = Math.floor(verticalPadding / 2);
  const paddingBottom = verticalPadding - paddingTop;

  const background = {
    r: PADDING_VALUE,
    g: PADDING_VALUE,
    b: PADDING_VALUE,
  };

  const { data, info } = await sharp(input, {
    failOn: "error",
  })
    .autoOrient()
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
    info.channels !== 3
  ) {
    throw new Error(
      `Unexpected letterbox output: ${info.width}x${info.height}x${info.channels}`,
    );
  }

  return {
    pixels: data,
    transform: {
      originalWidth,
      originalHeight,
      resizedWidth,
      resizedHeight,
      scale,
      paddingLeft,
      paddingTop,
      paddingRight,
      paddingBottom,
    },
  };
}
