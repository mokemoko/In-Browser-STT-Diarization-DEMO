import { defineConfig } from 'vite';

// onnxruntime-web のマルチスレッド WASM には cross-origin isolation (SharedArrayBuffer) が必要
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // GitHub Pages (https://<user>.github.io/<repo>/) でも動くよう相対パスで出力し、docs/ から配信する
  base: './',
  build: { outDir: 'docs' },
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web', '@huggingface/transformers'] },
});
