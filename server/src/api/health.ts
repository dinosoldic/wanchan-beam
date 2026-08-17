import type { FastifyPluginAsync } from "fastify";

const healthRoute: FastifyPluginAsync = async (app) => {
  app.get("/health", async () => {
    return {
      status: "ok",
    };
  });
};

export default healthRoute;
