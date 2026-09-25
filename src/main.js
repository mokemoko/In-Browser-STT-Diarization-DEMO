import './style.css';
import { LATENCY_PRESETS, NUM_SPEAKERS } from './diarization/session.js';
import { SAMPLE_RATE } from './diarization/features.js';
import { findSpeech, toSegments, toTurns } from './align.js';
import { startMicrophone, encodeWav } from './mic.js';

const DIAR_PUSH_SEC = 10; // 話者分離 worker に一度に渡す長さ
const ASR_WINDOW_SEC = 30; // Whisper の入力窓
// マイク入力時の文字起こし: 未確定区間を ASR_STEP_SEC ごとに Whisper にかけ直し、
// 連続する 2 回の結果で先頭が一致した単語を確定する。ASR_FORCE_SEC を超えたら最後の 1 語以外を確定する
const ASR_STEP_SEC = 1;
const ASR_FORCE_SEC = 20;
// Whisper は無音からも文章を作ってしまうので、話者分離で誰も話していない区間は渡さない。前後にこれだけ余白を残す
const SPEECH_PAD_SEC = 0.3;
const LABEL_WIDTH = 64; // タイムライン左の話者名の幅 (px)
const SPEAKER_COLORS = ['#3b6ef5', '#e5484d', '#30a46c', '#f5a524', '#8e4ec6', '#12a594', '#d6409f', '#7c8594'];

const $ = id => document.getElementById(id);
const ui = {
  env: $('env'),
  drop: $('drop'),
  dropLabel: $('drop-label'),
  file: $('file'),
  asrModel: $('asr-model'),
  language: $('language'),
  diarModel: $('diar-model'),
  preset: $('preset'),
  run: $('run'),
  mic: $('mic'),
  status: $('status'),
  diarText: $('diar-text'),
  diarProgress: $('diar-progress'),
  asrText: $('asr-text'),
  asrProgress: $('asr-progress'),
  result: $('result'),
  player: $('player'),
  timeline: $('timeline'),
  summary: $('summary'),
  transcript: $('transcript'),
};

// ---------- worker ----------

// 1 worker につき同時に 1 リクエストだけ投げる前提の小さなラッパー
function createWorkerClient(worker, onProgress) {
  let pending = null;
  worker.onmessage = ({ data }) => {
    if (data.type === 'progress') return onProgress(data);
    if (data.type === 'error') pending?.reject(new Error(data.text));
    else pending?.resolve(data);
    pending = null;
  };
  worker.onerror = e => {
    pending?.reject(new Error(e.message));
    pending = null;
  };
  return (message, transfer = []) =>
    new Promise((resolve, reject) => {
      pending = { resolve, reject };
      worker.postMessage(message, transfer);
    });
}

const diarWorker = new Worker(new URL('./workers/diarization.worker.js', import.meta.url), { type: 'module' });
const asrWorker = new Worker(new URL('./workers/asr.worker.js', import.meta.url), { type: 'module' });
const diarRequest = createWorkerClient(diarWorker, p => setTask('diar', p.text, p.progress));
const asrRequest = createWorkerClient(asrWorker, p => setTask('asr', p.text, p.progress));

// ---------- state ----------

let file = null;
let audio = null; // Float32Array 16 kHz mono (マイク入力中は audioBuffer の先頭部分のビュー)
let audioBuffer = new Float32Array(0);
let dataWaiters = [];
let recording = false;
let diarizing = false;
let stopMicrophone = null;
let probs = new Float32Array(0); // 話者確率 [T * 8]
let turns = [];
let hasWebGPU = false;

// ---------- setup ----------

async function detectEnvironment() {
  hasWebGPU = !!(navigator.gpu && (await navigator.gpu.requestAdapter().catch(() => null)));
  const badges = [
    [hasWebGPU ? 'WebGPU 利用可' : 'WebGPU なし（WASM で実行）', hasWebGPU],
    [crossOriginIsolated ? 'マルチスレッド WASM' : 'シングルスレッド WASM', crossOriginIsolated],
  ];
  ui.env.innerHTML = badges.map(([text, ok]) => `<span class="badge ${ok ? 'ok' : 'warn'}">${text}</span>`).join('');
  for (const option of ui.asrModel.querySelectorAll('[data-webgpu-only]')) option.disabled = !hasWebGPU;
}

for (const [name, preset] of Object.entries(LATENCY_PRESETS)) ui.preset.add(new Option(preset.label, name));

function selectFile(f) {
  if (!f) return;
  file = f;
  ui.dropLabel.textContent = `${f.name}（${(f.size / 1e6).toFixed(1)} MB）`;
  ui.run.disabled = false;
}

ui.file.addEventListener('change', () => selectFile(ui.file.files[0]));
ui.drop.addEventListener('dragover', e => {
  e.preventDefault();
  ui.drop.classList.add('over');
});
ui.drop.addEventListener('dragleave', () => ui.drop.classList.remove('over'));
ui.drop.addEventListener('drop', e => {
  e.preventDefault();
  ui.drop.classList.remove('over');
  selectFile(e.dataTransfer.files[0]);
});

// ---------- pipeline ----------

async function decodeAudio(f) {
  const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
  try {
    const buffer = await ctx.decodeAudioData(await f.arrayBuffer());
    const mono = new Float32Array(buffer.length);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
      const channel = buffer.getChannelData(c);
      for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / buffer.numberOfChannels;
    }
    return mono;
  } finally {
    ctx.close();
  }
}

function appendProbs(chunk) {
  const merged = new Float32Array(probs.length + chunk.length);
  merged.set(probs);
  merged.set(chunk, probs.length);
  probs = merged;
}

async function loadDiarization() {
  setTask('diar', 'モデルを準備中…', null);
  const ready = await diarRequest({ type: 'load', model: ui.diarModel.value });
  await diarRequest({ type: 'start', preset: ui.preset.value });
  return ready;
}

async function loadAsr() {
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  setTask('asr', 'モデルを準備中…', null);
  await asrRequest({ type: 'load', model: ui.asrModel.value, device });
  return device;
}

function resetResult() {
  ui.run.disabled = true;
  ui.mic.disabled = true;
  ui.status.hidden = false;
  ui.result.hidden = false;
  ui.player.hidden = false;
  probs = new Float32Array(0);
  renderTranscript([]);
}

ui.run.addEventListener('click', async () => {
  resetResult();
  try {
    setTask('diar', '音声をデコード中…', null);
    setTask('asr', '音声をデコード中…', null);
    audio = await decodeAudio(file);
    ui.player.src = URL.createObjectURL(file);
    drawTimeline();
    const [, words] = await processStream();
    renderTranscript(toTurns(words, probs));
  } catch (e) {
    console.error(e);
    alert(`エラー: ${e.message}`);
  } finally {
    ui.run.disabled = false;
    ui.mic.disabled = false;
  }
});

// ---------- microphone ----------

function appendAudio(samples) {
  const length = audio.length + samples.length;
  if (length > audioBuffer.length) {
    const grown = new Float32Array(Math.max(length, audioBuffer.length * 2, 60 * SAMPLE_RATE));
    grown.set(audio);
    audioBuffer = grown;
  }
  audioBuffer.set(samples, audio.length);
  audio = audioBuffer.subarray(0, length);
  notifyData();
}

// 新しい音声か話者確率が届くか、録音・話者分離が終わるまで待つ
const waitForData = () => new Promise(resolve => dataWaiters.push(resolve));

function notifyData() {
  const waiters = dataWaiters;
  dataWaiters = [];
  waiters.forEach(resolve => resolve());
}

async function stopRecording() {
  if (!recording) return;
  recording = false;
  ui.mic.disabled = true;
  ui.mic.classList.remove('recording');
  ui.mic.textContent = '残りを処理中…';
  await stopMicrophone?.();
  stopMicrophone = null;
  notifyData();
}

// 話者分離: 届いた分を DIAR_PUSH_SEC ずつ push する。session 側がチャンク + lookahead 分たまるまで確率を返さないだけ
async function diarizeStream({ threads }) {
  const live = recording;
  const started = performance.now();
  try {
    let pushed = 0;
    while (recording || pushed < audio.length) {
      if (pushed === audio.length) {
        await waitForData();
        continue;
      }
      // 転送で audio 本体が detach されないようコピーを渡す
      const samples = audio.slice(pushed, pushed + DIAR_PUSH_SEC * SAMPLE_RATE);
      pushed += samples.length;
      appendProbs((await diarRequest({ type: 'push', samples }, [samples.buffer])).probs);
      notifyData();
      if (recording) setTask('diar', `話者分離中…（遅れ ${Math.max(0, audio.length / SAMPLE_RATE - diarizedSec()).toFixed(1)} 秒）`, null);
      else setTask('diar', '話者分離中…', pushed / audio.length);
      drawTimeline();
    }
    appendProbs((await diarRequest({ type: 'finish' })).probs);
  } finally {
    diarizing = false;
    notifyData();
  }
  const sec = (performance.now() - started) / 1000;
  if (live) setTask('diar', `完了（${threads} スレッド）`, 1);
  else setTask('diar', `完了（${sec.toFixed(1)} 秒 / 実時間の ${(audio.length / SAMPLE_RATE / sec).toFixed(0)} 倍速・${threads} スレッド）`, 1);
}

const diarizedSec = () => probs.length / NUM_SPEAKERS / 100;
const sameWord = (a, b) => a.text.trim() === b.text.trim();
const inSpeech = w => findSpeech(probs, w.start - SPEECH_PAD_SEC, w.end + SPEECH_PAD_SEC) >= 0;

// 文字起こし: 未確定区間の先頭から最大 30 秒を Whisper に渡し、確定した単語の終わりから次の窓を始める。
// 話者分離で無音と分かっている区間は飛ばす。
// 録音中は低遅延のため ASR_STEP_SEC ごとに同じ区間をかけ直し、2 回続けて一致した単語を確定する。
// ファイル (と録音停止後) は窓全体の話者分離を待ってから 1 回だけかける
async function transcribeStream(device) {
  const live = recording;
  const started = performance.now();
  const committed = [];
  let tentative = []; // 前回の結果のうち未確定の単語
  let from = 0; // 未確定区間の先頭 (サンプル)
  let transcribedTo = 0;
  for (;;) {
    const done = !recording && !diarizing; // これ以上音声も話者確率も増えない
    const fromSec = from / SAMPLE_RATE;
    const diarized = Math.min(diarizedSec(), audio.length / SAMPLE_RATE);
    const speech = findSpeech(probs, fromSec, diarized);
    const skipTo = (speech < 0 ? diarized : speech) - SPEECH_PAD_SEC;
    if (skipTo > fromSec) {
      from = Math.round(skipTo * SAMPLE_RATE);
      tentative = []; // 無音と判定された区間の仮結果は捨てる
    }

    const end = Math.min(audio.length, from + ASR_WINDOW_SEC * SAMPLE_RATE);
    const ready = recording ? end - transcribedTo >= ASR_STEP_SEC * SAMPLE_RATE : done || diarized >= end / SAMPLE_RATE;
    if (speech < 0 || !ready) {
      if (done) break;
      await waitForData();
      continue;
    }
    transcribedTo = end;
    const start = from;
    const res = await asrRequest({
      type: 'transcribe',
      audio: audio.slice(start, end),
      offset: start / SAMPLE_RATE,
      language: ui.language.value,
    });
    // Whisper は内部で 30 秒にゼロ埋めするので、窓の外の時刻が返ることがある。窓に収め、無音区間の単語は落とす
    const windowEnd = end / SAMPLE_RATE;
    const words = res.words
      .map(w => ({ ...w, start: Math.min(w.start, windowEnd), end: Math.min(w.end, windowEnd) }))
      .filter(w => w.start < windowEnd)
      .filter(w => w.end > diarizedSec() || inSpeech(w));

    const isLast = !recording && end === audio.length;
    const flush = !recording || end - start >= ASR_FORCE_SEC * SAMPLE_RATE;
    let n = 0;
    while (n < tentative.length && n < words.length && sameWord(tentative[n], words[n])) n++;
    if (isLast) n = words.length;
    // 窓の終わりにかかる最後の単語は切れている可能性があるので、次の窓でかけ直す
    else if (flush) n = Math.max(n, words.at(-1)?.end < windowEnd - ASR_STEP_SEC ? words.length : words.length - 1);

    committed.push(...words.slice(0, n));
    tentative = words.slice(n);
    if (isLast) from = end;
    else if (n > 0) from = Math.min(end, Math.max(start, Math.round(words[n - 1].end * SAMPLE_RATE)));
    // 確定できる単語がない・時刻が進まないまま窓が長くなったときは、直近だけ残して先へ進む
    if (flush && !isLast && from <= start) from = end - ASR_STEP_SEC * SAMPLE_RATE;

    if (recording) setTask('asr', `文字起こし中…（${device}・未確定 ${((audio.length - from) / SAMPLE_RATE).toFixed(1)} 秒）`, null);
    else setTask('asr', `文字起こし中…（${device}）`, from / audio.length);
    renderTranscript(toTurns([...committed, ...tentative], probs));
    if (isLast) break;
  }
  const sec = (performance.now() - started) / 1000;
  if (live) setTask('asr', `完了（${device}）`, 1);
  else setTask('asr', `完了（${sec.toFixed(1)} 秒 / 実時間の ${(audio.length / SAMPLE_RATE / sec).toFixed(1)} 倍速・${device}）`, 1);
  // 確定時にまだ話者分離が済んでいなかった単語も、最終的な確率で無音なら落とす
  return committed.filter(inSpeech);
}

// 話者分離と文字起こしを並行して進める。モデルの準備ができた方から始める
function processStream() {
  diarizing = true;
  return Promise.all([loadDiarization().then(diarizeStream), loadAsr().then(transcribeStream)]).catch(e => {
    diarizing = false;
    notifyData();
    throw e;
  });
}

ui.mic.addEventListener('click', async () => {
  if (recording) return stopRecording();
  resetResult();
  ui.player.hidden = true;
  ui.player.removeAttribute('src');
  // オフライン設定 (30 秒遅れ) はマイクには向かないので、低遅延に切り替える
  if (ui.preset.value === 'offline') ui.preset.value = 'low_latency';
  try {
    // モデルを読み込み終えてから録音を始める (読み込み中の音声が溜まって遅れないように)
    await Promise.all([diarRequest({ type: 'load', model: ui.diarModel.value }), loadAsr()]);
    audio = new Float32Array(0);
    recording = true;
    stopMicrophone = await startMicrophone(appendAudio);
    ui.mic.textContent = '録音を停止';
    ui.mic.classList.add('recording');
    ui.mic.disabled = false;
    const [, words] = await processStream();
    renderTranscript(toTurns(words, probs));
    ui.player.src = URL.createObjectURL(encodeWav(audio));
    ui.player.hidden = false;
    drawTimeline();
  } catch (e) {
    console.error(e);
    alert(`エラー: ${e.message}`);
    await stopRecording();
  } finally {
    ui.run.disabled = !file;
    ui.mic.disabled = false;
    ui.mic.classList.remove('recording');
    ui.mic.textContent = 'マイクで録音を開始';
  }
});

// ---------- rendering ----------

function setTask(task, text, progress) {
  ui[`${task}Text`].textContent = text;
  const bar = ui[`${task}Progress`];
  if (progress == null) bar.removeAttribute('value');
  else bar.value = progress;
}

const formatTime = sec => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
};

const speakerName = s => (s < 0 ? '不明' : `話者 ${s + 1}`);

function activeSpeakers() {
  return [...new Set(toSegments(probs).map(seg => seg.speaker))].sort((a, b) => a - b);
}

function drawTimeline() {
  if (!audio?.length) return;
  const canvas = ui.timeline;
  const speakers = activeSpeakers();
  const rowHeight = 22;
  const width = canvas.clientWidth;
  const height = Math.max(1, speakers.length) * rowHeight + 20;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.height = `${height}px`;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  const style = getComputedStyle(document.documentElement);
  const duration = audio.length / SAMPLE_RATE;
  const x = sec => LABEL_WIDTH + (sec / duration) * (width - LABEL_WIDTH);

  g.font = '12px system-ui, sans-serif';
  g.textBaseline = 'middle';
  speakers.forEach((s, row) => {
    const y = row * rowHeight;
    g.fillStyle = style.getPropertyValue('--muted');
    g.fillText(speakerName(s), 0, y + rowHeight / 2);
    g.fillStyle = style.getPropertyValue('--border');
    g.fillRect(LABEL_WIDTH, y + rowHeight / 2 - 1, width - LABEL_WIDTH, 2);
  });
  for (const seg of toSegments(probs)) {
    g.fillStyle = SPEAKER_COLORS[seg.speaker];
    g.fillRect(x(seg.start), speakers.indexOf(seg.speaker) * rowHeight + 4, Math.max(1, x(seg.end) - x(seg.start)), rowHeight - 8);
  }
  // 話者分離の処理済み位置
  const processed = diarizedSec();
  if (processed < duration) {
    g.fillStyle = style.getPropertyValue('--muted');
    g.globalAlpha = 0.15;
    g.fillRect(x(processed), 0, width - x(processed), height - 20);
    g.globalAlpha = 1;
  }
  // 再生位置
  if (!recording) {
    g.fillStyle = style.getPropertyValue('--text');
    g.fillRect(x(ui.player.currentTime), 0, 2, height - 20);
  }
  g.fillStyle = style.getPropertyValue('--muted');
  g.textAlign = 'right';
  g.fillText(formatTime(duration), width, height - 8);
  g.textAlign = 'left';
  g.fillText('0:00', LABEL_WIDTH, height - 8);

  ui.summary.textContent = speakers.length ? `検出された話者: ${speakers.length} 人` : '';
}

function renderTranscript(newTurns) {
  turns = newTurns;
  ui.transcript.replaceChildren(
    ...turns.map(turn => {
      const li = document.createElement('li');
      const color = turn.speaker < 0 ? SPEAKER_COLORS[7] : SPEAKER_COLORS[turn.speaker];
      li.innerHTML = `
        <div class="speaker">
          <span class="chip" style="background:${color}">${speakerName(turn.speaker)}</span>
          <button class="time">${formatTime(turn.start)}</button>
        </div>
        <div class="text"></div>`;
      li.querySelector('.text').textContent = turn.text;
      li.querySelector('.time').addEventListener('click', () => seek(turn.start));
      return li;
    }),
  );
}

function seek(sec) {
  if (!ui.player.src) return;
  ui.player.currentTime = sec;
  ui.player.play();
}

ui.timeline.addEventListener('click', e => {
  if (!audio?.length) return;
  const rect = ui.timeline.getBoundingClientRect();
  const ratio = (e.clientX - rect.left - LABEL_WIDTH) / (rect.width - LABEL_WIDTH);
  if (ratio >= 0) seek(ratio * (audio.length / SAMPLE_RATE));
});

ui.player.addEventListener('timeupdate', () => {
  drawTimeline();
  const t = ui.player.currentTime;
  [...ui.transcript.children].forEach((li, i) => li.classList.toggle('active', turns[i].start <= t && t < turns[i].end));
});
window.addEventListener('resize', drawTimeline);

detectEnvironment();
