import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import Fastify, { type FastifyInstance } from "fastify";
import { readFile } from "node:fs/promises";

import { detectRoute, healthRoute } from "./api/index.js";
import { loadDetector } from "./inference/index.js";

export interface BuildAppOptions {
  logger?: boolean;
}

const sharedSchemaUrls = [
  new URL(
    "../../shared/schemas/dog-detection-response.schema.json",
    import.meta.url,
  ),
  new URL("../../shared/schemas/error-response.schema.json", import.meta.url),
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
    logger: options.logger ?? true,
  });

  await app.register(cors, {
    origin: getCorsOrigin(),
  });

  const sharedSchemas = await loadSharedSchemas();

  for (const schema of sharedSchemas) {
    app.addSchema(schema);
  }

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

function getCorsOrigin(): boolean | string[] {
  const configuredOrigins = (process.env.CORS_ORIGIN ?? "")
    .split(",")
    .map((origin) => origin.trim())
    .filter((origin) => origin.length > 0);

  if (configuredOrigins.length > 0) {
    return configuredOrigins;
  }

  return process.env.NODE_ENV !== "production";
}
