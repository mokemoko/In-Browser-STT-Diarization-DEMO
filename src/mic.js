// マイク入力と、録音した音声の WAV 化。

import { SAMPLE_RATE } from './diarization/features.js';
import workletUrl from './capture.worklet.js?url&no-inline';

/**
 * マイクを開き、16 kHz mono の Float32Array (約 100 ms ずつ) を onSamples に渡す。
 * @returns {Promise<() => Promise<void>>} 停止関数
 */
export async function startMicrophone(onSamples) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } });
  const ctx = new AudioContext();
  try {
    await ctx.audioWorklet.addModule(workletUrl);
    const node = new AudioWorkletNode(ctx, 'capture', { processorOptions: { targetRate: SAMPLE_RATE } });
    node.port.onmessage = ({ data }) => onSamples(data);
    // 出力は無音だが、destination につないでおかないと process() が呼ばれないブラウザがある
    ctx.createMediaStreamSource(stream).connect(node).connect(ctx.destination);
    await ctx.resume();
    return async () => {
      node.port.onmessage = null;
      stream.getTracks().forEach(t => t.stop());
      await ctx.close();
    };
  } catch (e) {
    stream.getTracks().forEach(t => t.stop());
    ctx.close();
    throw e;
  }
}

/** 16 kHz mono の Float32Array を 16 bit PCM の WAV にする */
export function encodeWav(samples) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const writeString = (offset, s) => [...s].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt チャンクの長さ
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 0x7fff, true);
  return new Blob([view], { type: 'audio/wav' });
}
