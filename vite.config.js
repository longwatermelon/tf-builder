import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // honor an externally assigned port so preview tooling can pick a free one
  server: process.env.PORT ? { port: Number(process.env.PORT), strictPort: true } : {},
})
