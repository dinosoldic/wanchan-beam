import assert from "node:assert/strict";
import { after, test } from "node:test";

import { buildApp } from "../src/app.js";

const app = await buildApp({
  logger: false,
});

after(async () => {
  await app.close();
});

test("GET /health returns the server status", async () => {
  const response = await app.inject({
    method: "GET",
    url: "/health",
  });

  assert.equal(response.statusCode, 200);
  assert.deepEqual(response.json(), {
    status: "ok",
  });
});
