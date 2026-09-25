import './style.css';
import { LATENCY_PRESETS, NUM_SPEAKERS } from './diarization/session.js';
import { SAMPLE_RATE } from './diarization/features.js';
import { toSegments, toTurns } from './align.js';

const DIAR_PUSH_SEC = 10; // 話者分離 worker に一度に渡す長さ
const ASR_WINDOW_SEC = 30; // Whisper の入力窓
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
let audio = null; // Float32Array 16 kHz mono
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

async function diarize() {
  setTask('diar', 'モデルを準備中…', null);
  const ready = await diarRequest({ type: 'load', model: ui.diarModel.value });
  await diarRequest({ type: 'start', preset: ui.preset.value });
  const started = performance.now();
  const step = DIAR_PUSH_SEC * SAMPLE_RATE;
  for (let i = 0; i < audio.length; i += step) {
    // 転送で audio 本体が detach されないようコピーを渡す
    const samples = audio.slice(i, i + step);
    const res = await diarRequest({ type: 'push', samples }, [samples.buffer]);
    appendProbs(res.probs);
    setTask('diar', '話者分離中…', Math.min(1, (i + step) / audio.length));
    drawTimeline();
  }
  appendProbs((await diarRequest({ type: 'finish' })).probs);
  const sec = (performance.now() - started) / 1000;
  setTask('diar', `完了（${sec.toFixed(1)} 秒 / 実時間の ${(audio.length / SAMPLE_RATE / sec).toFixed(0)} 倍速・${ready.threads} スレッド）`, 1);
}

async function transcribe() {
  const device = hasWebGPU ? 'webgpu' : 'wasm';
  setTask('asr', 'モデルを準備中…', null);
  await asrRequest({ type: 'load', model: ui.asrModel.value, device });
  const started = performance.now();
  const step = ASR_WINDOW_SEC * SAMPLE_RATE;
  const words = [];
  for (let i = 0; i < audio.length; i += step) {
    const res = await asrRequest({
      type: 'transcribe',
      audio: audio.slice(i, i + step),
      offset: i / SAMPLE_RATE,
      language: ui.language.value,
    });
    words.push(...res.words);
    setTask('asr', `文字起こし中…（${device}）`, Math.min(1, (i + step) / audio.length));
    // 話者分離が先に進んでいれば、途中経過も話者付きで表示する
    renderTranscript(toTurns(words, probs));
  }
  const sec = (performance.now() - started) / 1000;
  setTask('asr', `完了（${sec.toFixed(1)} 秒 / 実時間の ${(audio.length / SAMPLE_RATE / sec).toFixed(1)} 倍速・${device}）`, 1);
  return words;
}

ui.run.addEventListener('click', async () => {
  ui.run.disabled = true;
  ui.status.hidden = false;
  ui.result.hidden = false;
  probs = new Float32Array(0);
  renderTranscript([]);
  try {
    setTask('diar', '音声をデコード中…', null);
    setTask('asr', '音声をデコード中…', null);
    audio = await decodeAudio(file);
    ui.player.src = URL.createObjectURL(file);
    drawTimeline();
    const [, words] = await Promise.all([diarize(), transcribe()]);
    renderTranscript(toTurns(words, probs));
  } catch (e) {
    console.error(e);
    alert(`エラー: ${e.message}`);
  } finally {
    ui.run.disabled = false;
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
  if (!audio) return;
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
  const processed = probs.length / NUM_SPEAKERS / 100;
  if (processed < duration) {
    g.fillStyle = style.getPropertyValue('--muted');
    g.globalAlpha = 0.15;
    g.fillRect(x(processed), 0, width - x(processed), height - 20);
    g.globalAlpha = 1;
  }
  // 再生位置
  g.fillStyle = style.getPropertyValue('--text');
  g.fillRect(x(ui.player.currentTime), 0, 2, height - 20);
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
  ui.player.currentTime = sec;
  ui.player.play();
}

ui.timeline.addEventListener('click', e => {
  if (!audio) return;
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
