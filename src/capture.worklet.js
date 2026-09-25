// マイク音声を mono・targetRate に間引き、100 ms ごとにメインスレッドへ Float32Array で送る AudioWorklet。
// AudioContext のサンプルレート指定は MediaStream との組み合わせで動かないブラウザがあるので、ここで変換する。
// 区間平均で間引く (簡易なローパスを兼ねる) ので、44.1 kHz のような整数比でないレートでもよい。

class CaptureProcessor extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.ratio = sampleRate / processorOptions.targetRate;
    this.out = new Float32Array(Math.round(processorOptions.targetRate / 10));
    this.length = 0;
    this.sum = 0;
    this.count = 0;
    this.phase = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels?.length) return true;
    for (let i = 0; i < channels[0].length; i++) {
      let x = 0;
      for (const c of channels) x += c[i];
      this.sum += x / channels.length;
      this.count++;
      if (++this.phase < this.ratio) continue;
      this.phase -= this.ratio;
      this.out[this.length++] = this.sum / this.count;
      this.sum = 0;
      this.count = 0;
      if (this.length === this.out.length) {
        this.port.postMessage(this.out.slice());
        this.length = 0;
      }
    }
    return true;
  }
}

registerProcessor('capture', CaptureProcessor);
