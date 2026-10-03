import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Production build -> ui/web/dist, served by ui/server.py at "/" and "/assets".
// `npm run dev` proxies the API and the websocket to the Python backend on 8765.
export default defineConfig({
  plugins: [react()],
  base: '/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
});
