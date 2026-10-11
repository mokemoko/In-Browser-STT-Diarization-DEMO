// 文字起こし worker: Transformers.js + Whisper (単語タイムスタンプ付き ONNX) / Parakeet CTC (トークン単位の時刻)。
//   {type: 'load', model, device}                 -> {type: 'progress'} ... {type: 'ready'}
//   {type: 'transcribe', audio, offset, language} -> {type: 'words', words: [{text, start, end}]} (秒, 絶対時刻)
// 音声は 30 秒単位で受け取る。マイク入力でも同じ窓単位で送れる。
import { pipeline, env, AutoProcessor, AutoTokenizer, AutoModelForCTC } from '@huggingface/transformers';

env.allowLocalModels = false;

// Parakeet の CTC 出力は 1 フレーム 80 ms (hop 10 ms × subsampling 8)
const CTC_FRAME_SEC = 0.08;
const PARAKEET_DTYPES = { webgpu: 'q4', wasm: 'int8' };

// device ごとの量子化設定。large-v3-turbo は WebGPU 専用
const DTYPES = {
  webgpu: { encoder_model: 'fp32', decoder_model_merged: 'q4' },
  wasm: { encoder_model: 'q8', decoder_model_merged: 'q8' },
  'webgpu-large': { encoder_model: 'q4', decoder_model_merged: 'q4f16' },
};

let asr = null;
let loadedKey = null;

// Transformers.js の ASR pipeline は Parakeet の時刻を返さないので、logits の argmax から自前で作る
async function loadParakeet(model, device, progress_callback) {
  const [processor, tokenizer, ctc] = await Promise.all([
    AutoProcessor.from_pretrained(model),
    AutoTokenizer.from_pretrained(model),
    AutoModelForCTC.from_pretrained(model, { device, dtype: PARAKEET_DTYPES[device], progress_callback }),
  ]);
  const blank = ctc.config.pad_token_id;
  const vocab = [];
  for (const [token, id] of tokenizer.get_vocab()) vocab[id] = token.replaceAll('▁', ' ');
  const fn = async audio => {
    const { logits } = await ctc(await processor(audio));
    const [, frames, size] = logits.dims;
    const data = logits.data;
    const chunks = [];
    let prev = blank;
    for (let t = 0; t < frames; t++) {
      let id = 0;
      for (let v = 1; v < size; v++) if (data[t * size + v] > data[t * size + id]) id = v;
      if (id !== blank && id !== prev) {
        chunks.push({ text: vocab[id], timestamp: [t * CTC_FRAME_SEC, (t + 1) * CTC_FRAME_SEC] });
      } else if (id !== blank) {
        chunks.at(-1).timestamp[1] = (t + 1) * CTC_FRAME_SEC;
      }
      prev = id;
    }
    return { chunks };
  };
  fn.dispose = () => ctc.dispose();
  return fn;
}

async function load(model, device) {
  const key = `${model}|${device}`;
  if (key === loadedKey) return;
  await asr?.dispose();
  asr = null;
  loadedKey = null;
  const files = {};
  const progress_callback = p => {
    if (p.status !== 'progress' || !p.total) return;
    files[p.file] = [p.loaded, p.total];
    const [loaded, total] = Object.values(files).reduce((a, [l, t]) => [a[0] + l, a[1] + t], [0, 0]);
    postMessage({ type: 'progress', text: `モデルをダウンロード中 ${(loaded / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`, progress: loaded / total });
  };
  if (model.includes('parakeet')) {
    asr = await loadParakeet(model, device, progress_callback);
  } else {
    const dtype = device === 'webgpu' && model.includes('large') ? DTYPES['webgpu-large'] : DTYPES[device];
    asr = await pipeline('automatic-speech-recognition', model, { device, dtype, progress_callback });
  }
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
