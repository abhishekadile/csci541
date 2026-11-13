import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// Development: Vite on :5173 proxies /api to the FastAPI server on :8000 (same-origin from the browser's view).
// Production: FastAPI serves frontend/dist itself, so the frontend and API share one origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, ".", "");
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: { "/api": { target: env.VITE_API_TARGET || "http://127.0.0.1:8000", changeOrigin: false } },
    },
    build: { outDir: "dist", sourcemap: false },
  };
});
