import multipart from "@fastify/multipart";
import Fastify from "fastify";

import { healthRoute, detectRoute } from "./api/index.js";
import { loadDetector } from "./inference/index.js";

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? "0.0.0.0";

const app = Fastify({
  logger: true,
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

try {
  await app.listen({ port, host });
} catch (error) {
  app.log.error(error);
  process.exit(1);
}
