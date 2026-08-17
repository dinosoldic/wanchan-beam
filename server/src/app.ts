import multipart from "@fastify/multipart";
import Fastify, { type FastifyInstance } from "fastify";

import { detectRoute, healthRoute } from "./api/index.js";
import { loadDetector } from "./inference/index.js";

export interface BuildAppOptions {
  logger?: boolean;
}

export async function buildApp(
  options: BuildAppOptions = {},
): Promise<FastifyInstance> {
  const app = Fastify({
    logger: options.logger ?? true,
  });

  const detector = await loadDetector();

  app.log.info(
    {
      inputs: detector.inputNames,
      outputs: detector.outputNames,
    },
    "Detector loaded",
  );

  await app.register(multipart, {
    limits: {
      files: 1,
      fileSize: 10 * 1024 * 1024,
      parts: 1,
    },
  });

  await app.register(healthRoute);
  await app.register(detectRoute);

  return app;
}
