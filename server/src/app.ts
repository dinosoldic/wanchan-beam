import multipart from "@fastify/multipart";
import Fastify, { type FastifyInstance } from "fastify";
import { readFile } from "node:fs/promises";

import { detectRoute, healthRoute } from "./api/index.js";
import { loadDetector, loadBreedClassifier } from "./inference/index.js";

export interface BuildAppOptions {
  logger?: boolean;
}

const sharedSchemaUrls = [
  new URL("../schemas/dog-detection-response.schema.json", import.meta.url),
  new URL("../schemas/error-response.schema.json", import.meta.url),
];

async function loadSharedSchemas(): Promise<Record<string, unknown>[]> {
  return Promise.all(
    sharedSchemaUrls.map(async (schemaUrl) => {
      const contents = await readFile(schemaUrl, "utf8");

      return JSON.parse(contents) as Record<string, unknown>;
    }),
  );
}

export async function buildApp(
  options: BuildAppOptions = {},
): Promise<FastifyInstance> {
  const app = Fastify({
    logger: options.logger ?? {
      level: process.env.LOG_LEVEL ?? "warn",
    },
  });

  const sharedSchemas = await loadSharedSchemas();

  for (const schema of sharedSchemas) {
    app.addSchema(schema);
  }

  // load models
  await Promise.all([loadDetector(), loadBreedClassifier()]);

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
