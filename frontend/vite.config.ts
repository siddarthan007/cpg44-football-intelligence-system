import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API_TARGET = process.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        ws: false,
      },
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});
