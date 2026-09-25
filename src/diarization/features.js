// Nemotron の前処理 (NemotronAsrStreamingFeatureExtractor) をインクリメンタルに計算する。
// 16 kHz mono → pre-emphasis 0.97 → STFT (n_fft 512, 対称 Hann 400 を中央配置, hop 160, center=True)
// → パワースペクトル → Slaney mel 128 → log(x + 2^-24)。正規化はしない。
// push() で届いた分だけフレームを返すので、ファイルでもマイクでも同じように使える。

export const SAMPLE_RATE = 16000;
export const HOP = 160;
export const N_MELS = 128;
const N_FFT = 512;
const WIN = 400;
const PREEMPH = 0.97;
const HALF = N_FFT / 2;
const N_BINS = N_FFT / 2 + 1;
const LOG_GUARD = 2 ** -24;

function createFFT(n) {
  const bits = Math.log2(n);
  const rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r = (r << 1) | ((i >> b) & 1);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2);
  const sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = -Math.sin((2 * Math.PI * i) / n);
  }
  // in-place の radix-2 FFT
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const a = start + k;
          const b = a + half;
          const wr = cos[k * step];
          const wi = sin[k * step];
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  };
}

export class LogMelStream {
  /** @param {Float32Array} melFilters [128 * 257] (librosa, slaney) */
  constructor(melFilters) {
    this.filters = melFilters;
    // mel フィルタは疎なので、各バンドの非ゼロ区間だけ計算する
    this.bandLo = new Int32Array(N_MELS);
    this.bandHi = new Int32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let lo = N_BINS;
      let hi = 0;
      for (let k = 0; k < N_BINS; k++) {
        if (melFilters[m * N_BINS + k] !== 0) {
          lo = Math.min(lo, k);
          hi = k + 1;
        }
      }
      this.bandLo[m] = lo;
      this.bandHi[m] = hi;
    }
    // torch.stft は win_length の窓を n_fft の中央にゼロ埋めする
    this.window = new Float64Array(N_FFT);
    const offset = (N_FFT - WIN) / 2;
    for (let i = 0; i < WIN; i++) this.window[offset + i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (WIN - 1));
    this.fft = createFFT(N_FFT);
    this.re = new Float64Array(N_FFT);
    this.im = new Float64Array(N_FFT);

    this.buffer = new Float32Array(0); // pre-emphasis 済みサンプル
    this.bufferStart = 0; // buffer[0] の絶対サンプル位置
    this.numSamples = 0;
    this.prevSample = 0;
    this.nextFrame = 0;
  }

  /** 新しいサンプルを追加し、計算可能になったフレームを返す */
  push(samples) {
    const emph = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) {
      const x = samples[i];
      emph[i] = this.numSamples + i === 0 ? x : x - PREEMPH * this.prevSample;
      this.prevSample = x;
    }
    this.numSamples += samples.length;
    const merged = new Float32Array(this.buffer.length + emph.length);
    merged.set(this.buffer);
    merged.set(emph, this.buffer.length);
    this.buffer = merged;
    // フレーム f は [f*hop - 256, f*hop + 256) のサンプルを使う
    const available = Math.floor((this.numSamples - HALF) / HOP) + 1;
    return this.#emit(Math.max(this.nextFrame, available));
  }

  /** ストリーム終端。右端をゼロ埋めして floor(L / hop) フレームまで出す */
  finish() {
    return this.#emit(Math.floor(this.numSamples / HOP));
  }

  #emit(endFrame) {
    const count = Math.max(0, endFrame - this.nextFrame);
    const out = new Float32Array(count * N_MELS);
    const { re, im, window, filters } = this;
    for (let n = 0; n < count; n++) {
      const start = (this.nextFrame + n) * HOP - HALF;
      for (let i = 0; i < N_FFT; i++) {
        const idx = start + i;
        const x = idx >= 0 && idx < this.numSamples ? this.buffer[idx - this.bufferStart] : 0;
        re[i] = x * window[i];
        im[i] = 0;
      }
      this.fft(re, im);
      const base = n * N_MELS;
      for (let m = 0; m < N_MELS; m++) {
        let acc = 0;
        const row = m * N_BINS;
        for (let k = this.bandLo[m]; k < this.bandHi[m]; k++) acc += filters[row + k] * (re[k] * re[k] + im[k] * im[k]);
        out[base + m] = Math.log(acc + LOG_GUARD);
      }
    }
    this.nextFrame += count;
    // 次のフレームで使わないサンプルを捨てる
    const keepFrom = Math.max(0, this.nextFrame * HOP - HALF);
    if (keepFrom > this.bufferStart) {
      this.buffer = this.buffer.slice(keepFrom - this.bufferStart);
      this.bufferStart = keepFrom;
    }
    return { data: out, numFrames: count };
  }
}
