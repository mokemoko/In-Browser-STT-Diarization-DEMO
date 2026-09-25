// Nemotron-3-Diarization のストリーミング推論セッション。
// transformers の Nemotron3DiarizationForAudioFrameClassification.forward のチャンクループを移植したもの。
// 音声を push() するたびに、確定した 10 ms フレームの話者確率 [n, 8] を返す。
// ファイルは「全体を順に push → finish()」、マイクは「届いた分を push」で同じコードを使う。

import { LogMelStream, N_MELS } from './features.js';
import { SpeakerCache } from './speakerCache.js';

export const NUM_SPEAKERS = 8;
const SUBSAMPLING = 8; // mel 8 フレーム = encoder 1 フレーム (80 ms)
const HIDDEN = 512;

// chunkLength / rightContext は encoder フレーム (80 ms) 単位。
// offline は config.json のトップレベル値、それ以外は processor_config.json の streaming_modes と streaming_config。
export const LATENCY_PRESETS = {
  offline: { label: '30.4 s（オフライン・高精度）', chunkLength: 340, rightContext: 40, fifoLength: 40, updatePeriod: 300 },
  low_latency: { label: '1.04 s（低遅延）', chunkLength: 9, rightContext: 4, fifoLength: 264, updatePeriod: 222 },
  very_low_latency: { label: '0.64 s', chunkLength: 6, rightContext: 2, fifoLength: 264, updatePeriod: 222 },
  ultra_low_latency: { label: '0.32 s（超低遅延）', chunkLength: 3, rightContext: 1, fifoLength: 264, updatePeriod: 222 },
};

const sigmoid = x => 1 / (1 + Math.exp(-x));

export class DiarizationSession {
  /**
   * @param {object} models { ort, embedSession, stepSession, melFilters, silenceEmbed }
   * @param {keyof LATENCY_PRESETS} presetName
   */
  constructor(models, presetName) {
    this.models = models;
    this.preset = LATENCY_PRESETS[presetName];
    this.mel = new LogMelStream(models.melFilters);
    this.cache = new SpeakerCache(NUM_SPEAKERS, this.preset.fifoLength, this.preset.updatePeriod, models.silenceEmbed);
    this.pendingMel = new Float32Array(0); // 8 フレームに満たず、まだ埋め込んでいない mel
    this.embeds = []; // 次のチャンク先頭以降の encoder 埋め込み
    this.numMelFrames = 0;
    this.numOutputFrames = 0;
  }

  /** @param {Float32Array} samples 16 kHz mono。戻り値は新たに確定した確率 Float32Array [n * 8] */
  async push(samples) {
    const mel = this.mel.push(samples);
    await this.#embed(mel.data, false);
    const { chunkLength, rightContext } = this.preset;
    const outputs = [];
    while (this.embeds.length >= chunkLength + rightContext) {
      outputs.push(await this.#step(chunkLength, rightContext));
    }
    const probs = concat(outputs);
    this.numOutputFrames += probs.length / NUM_SPEAKERS;
    return probs;
  }

  /** ストリーム終端。残りを lookahead なしで処理し、最後の確率を返す */
  async finish() {
    const mel = this.mel.finish();
    await this.#embed(mel.data, true);
    const { chunkLength, rightContext } = this.preset;
    const outputs = [];
    while (this.embeds.length > 0) {
      const n = Math.min(chunkLength, this.embeds.length);
      outputs.push(await this.#step(n, Math.min(rightContext, this.embeds.length - n)));
    }
    // 最後の encoder フレームは stacking のゼロ埋め分を含むので、実在する mel フレーム数に切り詰める
    const probs = concat(outputs).subarray(0, (this.numMelFrames - this.numOutputFrames) * NUM_SPEAKERS);
    this.numOutputFrames += probs.length / NUM_SPEAKERS;
    return probs;
  }

  // mel を 8 フレームずつまとめて embed.onnx に通す。埋め込みは自分の 8 フレームにしか依存しない
  async #embed(melData, isLast) {
    const { ort, embedSession } = this.models;
    this.numMelFrames += melData.length / N_MELS;
    let mel = new Float32Array(this.pendingMel.length + melData.length);
    mel.set(this.pendingMel);
    mel.set(melData, this.pendingMel.length);
    let numFrames = mel.length / N_MELS;
    if (isLast && numFrames % SUBSAMPLING !== 0) {
      // 元実装と同じく、最後の不完全なグループはゼロ埋めする
      const padded = new Float32Array(Math.ceil(numFrames / SUBSAMPLING) * SUBSAMPLING * N_MELS);
      padded.set(mel);
      mel = padded;
      numFrames = padded.length / N_MELS;
    }
    const usable = numFrames - (numFrames % SUBSAMPLING);
    this.pendingMel = mel.slice(usable * N_MELS);
    if (usable === 0) return;
    const result = await embedSession.run({
      features: new ort.Tensor('float32', mel.subarray(0, usable * N_MELS), [1, usable, N_MELS]),
    });
    const data = result.embeds.data;
    for (let i = 0; i < result.embeds.dims[1]; i++) this.embeds.push(data.slice(i * HIDDEN, (i + 1) * HIDDEN));
  }

  // 1 チャンク分: [speaker cache, FIFO, chunk, lookahead] を step.onnx に通す
  async #step(numChunkFrames, numLookahead) {
    const { ort, stepSession } = this.models;
    const cached = this.cache.getEmbeds();
    const input = cached.concat(this.embeds.slice(0, numChunkFrames + numLookahead));
    const T = input.length;
    const buffer = new Float32Array(T * HIDDEN);
    input.forEach((e, i) => buffer.set(e, i * HIDDEN));
    const result = await stepSession.run({ embeds: new ort.Tensor('float32', buffer, [1, T, HIDDEN]) });
    const logits = result.logits.data; // [T * 8, 8]

    // キャッシュ更新用に encoder フレームレートへ平均プーリング
    const pooled = [];
    for (let t = 0; t < T; t++) {
      const p = new Float32Array(NUM_SPEAKERS);
      for (let k = 0; k < SUBSAMPLING; k++) {
        const row = (t * SUBSAMPLING + k) * NUM_SPEAKERS;
        for (let s = 0; s < NUM_SPEAKERS; s++) p[s] += sigmoid(logits[row + s]) / SUBSAMPLING;
      }
      pooled.push(p);
    }
    this.cache.update(input, pooled, numChunkFrames);
    this.embeds = this.embeds.slice(numChunkFrames);

    const from = cached.length * SUBSAMPLING * NUM_SPEAKERS;
    const to = (cached.length + numChunkFrames) * SUBSAMPLING * NUM_SPEAKERS;
    const probs = new Float32Array(to - from);
    for (let i = 0; i < probs.length; i++) probs[i] = sigmoid(logits[from + i]);
    return probs;
  }
}

function concat(arrays) {
  const out = new Float32Array(arrays.reduce((n, a) => n + a.length, 0));
  let offset = 0;
  for (const a of arrays) {
    out.set(a, offset);
    offset += a.length;
  }
  return out;
}
