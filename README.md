# stt-diarization

ブラウザ内だけで文字起こし (STT) と話者分離を行うデモ。音声はサーバーへ送らない。

```sh
npm install
npm run dev   # http://localhost:5173
```

## GitHub Pages

```sh
npm run build   # docs/ に静的ファイルを出力
```

`docs/` をコミットし、リポジトリの Settings → Pages で「Deploy from a branch / main / docs」を選ぶ。

GitHub Pages は COOP/COEP ヘッダーを付けられないため、[coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) (MIT, `public/coi-serviceworker.js`) で付与してマルチスレッド WASM を有効にしている。初回アクセス時に 1 回だけ自動リロードが入る。

## 構成

| 処理 | モデル | ランタイム |
|---|---|---|
| 文字起こし | Whisper base / small / large-v3-turbo (`onnx-community/*_timestamped`, 単語タイムスタンプ付き) | Transformers.js (WebGPU があれば WebGPU, なければ WASM) |
| 話者分離 | NVIDIA Nemotron-3-Diarization の ONNX 変換版 ([NealCaren/Nemotron-3-Diarization-ONNX](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX)) | onnxruntime-web (WASM, マルチスレッド) |
| 突き合わせ | 各単語の区間で話者確率の平均が最大の話者を割り当て、同じ話者の連続を 1 発話にまとめる | plain JS |

Transformers.js 4.3.0 は `nemotron3_diarization` に未対応で、ONNX にはニューラルネット部分 (`embed.onnx` / `step.onnx`) しか入っていない。
そのため前処理・チャンク処理・話者キャッシュは transformers の Python 実装から `src/diarization/` に移植している。

- `features.js` — log-mel 特徴量 (NemotronAsrStreamingFeatureExtractor) をインクリメンタルに計算
- `speakerCache.js` — Arrival-Order Speaker Cache + FIFO (Nemotron3DiarizationSpeakerCache)
- `session.js` — チャンクループ (Nemotron3DiarizationForAudioFrameClassification.forward)。`push(samples)` で確定したフレームの話者確率を返し、`finish()` で残りを処理する

モデルは初回にダウンロードし、Cache Storage / ブラウザキャッシュに保存する。

## マイク入力 (リアルタイム) への拡張

話者分離は最初からストリーミング API になっていて、ファイル処理も 10 秒ずつ `push` しているだけ。
マイク対応では AudioWorklet で 16 kHz に変換したサンプルを、worker にそのまま `push` すればよい。

- レイテンシー設定は UI で選べる (`LATENCY_PRESETS`)。マイクなら `low_latency` (1.04 s) 以下を使う
- Whisper はストリーミングではないため、数秒〜30 秒の窓ごとに `transcribe` を送る形になる
- 途中結果の単語への話者割り当ては `toTurns(words, probs)` をその都度呼べばよい

## ライセンス

このリポジトリのコードは [MIT](LICENSE)。`src/diarization/` は Hugging Face transformers (Apache-2.0) の Nemotron3Diarization 実装を移植したもの。
モデルは同梱せず実行時に取得する: Nemotron-3-Diarization は NVIDIA の [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization)、Whisper は MIT。
