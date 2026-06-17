import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'

function loadHttpsConfig() {
  const certPath = process.env.FRONTEND_HTTPS_CERT_FILE
  const keyPath = process.env.FRONTEND_HTTPS_KEY_FILE
  if (!certPath || !keyPath || !existsSync(certPath) || !existsSync(keyPath)) {
    return undefined
  }
  return {
    cert: readFileSync(certPath),
    key: readFileSync(keyPath),
  }
}

export default defineConfig({
  plugins: [vue()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        inference: resolve(__dirname, 'inference.html'),
      },
    },
  },
  server: {
    port: 8000,
    strictPort: true,
    https: loadHttpsConfig(),
    proxy: {
      '/api': { target: 'http://localhost:8001', changeOrigin: true },
      '/reports': { target: 'http://localhost:8001', changeOrigin: true },
    },
  },
})
