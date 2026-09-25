// 文字起こし worker: Transformers.js + Whisper (単語タイムスタンプ付き ONNX)。
//   {type: 'load', model, device}                 -> {type: 'progress'} ... {type: 'ready'}
//   {type: 'transcribe', audio, offset, language} -> {type: 'words', words: [{text, start, end}]} (秒, 絶対時刻)
// 音声は 30 秒単位で受け取る。マイク入力でも同じ窓単位で送れる。
import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

// device ごとの量子化設定。large-v3-turbo は WebGPU 専用
const DTYPES = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm: { encoder_model: 'q8', decoder_model_merged: 'q8' },
  'webgpu-large': { encoder_model: 'q4', decoder_model_merged: 'q4f16' },
};

let asr = null;
let loadedKey = null;

async function load(model, device) {
  const key = `${model}|${device}`;
  if (key === loadedKey) return;
  await asr?.dispose();
  asr = null;
  loadedKey = null;
  const files = {};
  const dtype = device === 'webgpu' && model.includes('large') ? DTYPES['webgpu-large'] : DTYPES[device];
  asr = await pipeline('automatic-speech-recognition', model, {
    device,
    dtype,
    progress_callback: p => {
      if (p.status !== 'progress' || !p.total) return;
      files[p.file] = [p.loaded, p.total];
      const [loaded, total] = Object.values(files).reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
      postMessage({ type: 'progress', text: `モデルをダウンロード中 ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, progress: loaded / total });
    },
  });
  loadedKey = key;
}

onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      await load(data.model, data.device);
      postMessage({ type: 'ready' });
    } else if (data.type === 'transcribe') {
      const out = await asr(data.audio, {
        return_timestamps: 'word',
        language: data.language || null,
        task: 'transcribe',
      });
      const words = (out.chunks ?? []).map(c => ({
        text: c.text,
        start: data.offset + (c.timestamp[0] ?? 0),
        end: data.offset + (c.timestamp[1] ?? c.timestamp[0] ?? 0),
      }));
      postMessage({ type: 'words', words });
    }
  } catch (e) {
    postMessage({ type: 'error', text: String(e?.message ?? e) });
  }
};
