// 話者確率 (10 ms ごと [T, 8]) と Whisper の単語タイムスタンプを突き合わせる。

import { NUM_SPEAKERS } from './diarization/session.js';

const FRAMES_PER_SEC = 100;

/** 閾値を超えた区間を話者ごとの区間 [{speaker, start, end}] (秒) にする */
export function toSegments(probs, threshold = 0.5) {
  const numFrames = probs.length / NUM_SPEAKERS;
  const segments = [];
  for (let s = 0; s < NUM_SPEAKERS; s++) {
    let onset = -1;
    for (let t = 0; t <= numFrames; t++) {
      const active = t < numFrames && probs[t * NUM_SPEAKERS + s] > threshold;
      if (active && onset < 0) onset = t;
      if (!active && onset >= 0) {
        segments.push({ speaker: s, start: onset / FRAMES_PER_SEC, end: t / FRAMES_PER_SEC });
        onset = -1;
      }
    }
  }
  return segments.sort((a, b) => a.start - b.start);
}

// [start, end] 秒の区間で確率の平均が最大の話者と、その平均値
function dominantSpeaker(probs, start, end) {
  const numFrames = probs.length / NUM_SPEAKERS;
  const f0 = Math.max(0, Math.floor(start * FRAMES_PER_SEC));
  const f1 = Math.min(numFrames, Math.max(f0 + 1, Math.ceil(end * FRAMES_PER_SEC)));
  if (f1 <= f0) return { speaker: -1, mass: 0 };
  const sums = new Float64Array(NUM_SPEAKERS);
  for (let f = f0; f < f1; f++) for (let s = 0; s < NUM_SPEAKERS; s++) sums[s] += probs[f * NUM_SPEAKERS + s];
  let best = 0;
  for (let s = 1; s < NUM_SPEAKERS; s++) if (sums[s] > sums[best]) best = s;
  return { speaker: best, mass: sums[best] / (f1 - f0) };
}

/**
 * 単語ごとに話者を決め、同じ話者の連続を 1 つの発話にまとめる。
 * 単語区間が無音判定なら前後に広げて探し、それでも無ければ直前の話者を引き継ぐ。
 * @returns {{speaker: number, start: number, end: number, text: string}[]}
 */
export function toTurns(words, probs) {
  const turns = [];
  let previous = -1;
  for (const w of words) {
    let r = dominantSpeaker(probs, w.start, w.end);
    for (let pad = 0.25; r.mass < 0.2 && pad <= 1; pad *= 2) r = dominantSpeaker(probs, w.start - pad, w.end + pad);
    const speaker = r.mass < 0.05 && previous >= 0 ? previous : r.speaker;
    previous = speaker;

    const last = turns.at(-1);
    if (last && last.speaker === speaker) {
      last.text += w.text;
      last.end = w.end;
    } else {
      turns.push({ speaker, start: w.start, end: w.end, text: w.text });
    }
  }
  for (const t of turns) t.text = t.text.trim();
  return turns;
}
