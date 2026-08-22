export interface Size {
  width: number;
  height: number;
}

export interface ContainedImageLayout extends Size {
  top: number;
  left: number;
}

export function getContainedImageLayout(
  container: Size,
  image: Size | null,
): ContainedImageLayout | null {
  if (
    !image ||
    container.width <= 0 ||
    container.height <= 0 ||
    image.width <= 0 ||
    image.height <= 0
  ) {
    return null;
  }

  const scale = Math.min(
    container.width / image.width,
    container.height / image.height,
  );

  const width = image.width * scale;
  const height = image.height * scale;

  return {
    width,
    height,
    left: (container.width - width) / 2,
    top: (container.height - height) / 2,
  };
}
