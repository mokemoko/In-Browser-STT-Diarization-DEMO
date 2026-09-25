// 話者分離 worker: Nemotron-3-Diarization (ONNX) を onnxruntime-web (WASM) で動かす。
//   {type: 'load', model}            -> {type: 'progress'} ... {type: 'ready'}
//   {type: 'start', preset}          -> {type: 'started'}
//   {type: 'push', samples}          -> {type: 'probs', probs}   (新たに確定したフレーム [n * 8])
//   {type: 'finish'}                 -> {type: 'probs', probs, done: true}
import * as ort from 'onnxruntime-web/wasm';
import { DiarizationSession } from '../diarization/session.js';
// バンドルすると ORT がスレッド用の glue (.mjs) を見つけられず初期化が止まるので、別ファイルとして配信して明示する
import ortWasmUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import ortMjsUrl from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url';

const MODEL_BASE = 'https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX/resolve/main/';
const CACHE_NAME = 'nemotron-diarization-v1';

ort.env.wasm.wasmPaths = { wasm: ortWasmUrl, mjs: ortMjsUrl };
// マルチスレッドは cross-origin isolated のときだけ使える
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

let models = null;
let loadedModel = null;
let session = null;

// Cache Storage に保存して、2 回目以降はダウンロードしない
async function fetchCached(url, onProgress) {
  const cache = await caches.open(CACHE_NAME).catch(() => null);
  const hit = await cache?.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} の取得に失敗しました (HTTP ${res.status})`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.length;
  }
  await cache?.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

async function load(model) {
  if (loadedModel === model) return;
  const [embed, step, mel, silence] = await Promise.all([
    fetchCached(MODEL_BASE + 'embed.onnx'),
    fetchCached(MODEL_BASE + `${model}.onnx`, (loaded, total) =>
      postMessage({ type: 'progress', text: `モデルをダウンロード中 ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, progress: total ? loaded / total : null }),
    ),
    fetchCached(MODEL_BASE + 'mel_filters.bin'),
    fetchCached(MODEL_BASE + 'silence_embeds.bin'),
  ]);
  postMessage({ type: 'progress', text: 'モデルを初期化中…', progress: null });
  const options = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
  models = {
    ort,
    embedSession: await ort.InferenceSession.create(embed, options),
    stepSession: await ort.InferenceSession.create(step, options),
    melFilters: new Float32Array(mel.buffer),
    silenceEmbed: new Float32Array(silence.buffer),
  };
  loadedModel = model;
}

onmessage = async ({ data }) => {
  try {
    switch (data.type) {
      case 'load':
        await load(data.model);
        postMessage({ type: 'ready', threads: ort.env.wasm.numThreads });
        break;
      case 'start':
        session = new DiarizationSession(models, data.preset);
        postMessage({ type: 'started' });
        break;
      case 'push': {
        const probs = await session.push(data.samples);
        postMessage({ type: 'probs', probs }, [probs.buffer]);
        break;
      }
      case 'finish': {
        const probs = await session.finish();
        session = null;
        postMessage({ type: 'probs', probs, done: true }, [probs.buffer]);
        break;
      }
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e?.message ?? e) });
  }
};
