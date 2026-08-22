export class InvalidImageError extends Error {
  constructor(cause: unknown) {
    super("Uploaded file is not a valid supported image", {
      cause,
    });

    this.name = "InvalidImageError";
  }
}
