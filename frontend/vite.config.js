import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
// Dev proxy: /api/* forwards to the backend started by scripts/start.py
// (DW_PORT, default 8010). This keeps the browser decoupled from the
// backend port so a stale VITE_API_BASE_URL cannot break the app.
const backendTarget = `http://localhost:${process.env.DW_PORT || 8010}`;

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    proxy: {
      '/api': {
        target: backendTarget,
        changeOrigin: true,
      },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(import.meta.dirname, "./src"),
    },
  },
})
