import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Production build is served by FastAPI from frontend/dist/ at the site root.
  base: '/',
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
  // `npm run dev` proxies API + plot calls to the FastAPI backend on :7860.
  server: {
    port: 5173,
    proxy: {
      '/api':    { target: 'http://localhost:7860', changeOrigin: true },
      '/plots':  { target: 'http://localhost:7860', changeOrigin: true },
      '/health': { target: 'http://localhost:7860', changeOrigin: true },
    },
  },
})
