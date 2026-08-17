import type { FastifyPluginAsync } from "fastify";

import { detectDogs } from "../inference/index.js";
import { InvalidImageError } from "../preprocessing/index.js";

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
]);

const detectRoute: FastifyPluginAsync = async (app) => {
  app.post("/detect", async (request, reply) => {
    const uploadedImage = await request.file();

    if (!uploadedImage) {
      return reply.code(400).send({
        error: "An image file is required",
      });
    }

    const imageBuffer = await uploadedImage.toBuffer();

    if (uploadedImage.fieldname !== "image") {
      return reply.code(400).send({
        error: 'The uploaded file field must be named "image"',
      });
    }

    if (!SUPPORTED_IMAGE_TYPES.has(uploadedImage.mimetype)) {
      return reply.code(415).send({
        error: `Unsupported image type: ${uploadedImage.mimetype}`,
      });
    }

    try {
      const result = await detectDogs(imageBuffer);

      return reply.code(200).send(result);
    } catch (error) {
      if (error instanceof InvalidImageError) {
        return reply.code(422).send({
          error: error.message,
        });
      }

      throw error;
    }
  });
};

export default detectRoute;
