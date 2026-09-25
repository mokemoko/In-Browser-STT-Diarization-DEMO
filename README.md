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
| 話者分離 | NVIDIA Nemotron-3-Diarization の ONNX 変換版 ([NealCaren/Nemotron-3-Diarization-ONNX](https://huggingface.co/NealCaren/Nemotron-3-Diarization-ONNX)) | onnxruntime-web。INT8 を WASM (マルチスレッド) か、FP32 を WebGPU で。WebGPU があれば既定で WebGPU |
| 突き合わせ | 各単語の区間で話者確率の平均が最大の話者を割り当て、同じ話者の連続を 1 発話にまとめる | plain JS |

Transformers.js 4.3.0 は `nemotron3_diarization` に未対応で、ONNX にはニューラルネット部分 (`embed.onnx` / `step.onnx`) しか入っていない。
そのため前処理・チャンク処理・話者キャッシュは transformers の Python 実装から `src/diarization/` に移植している。

- `features.js` — log-mel 特徴量 (NemotronAsrStreamingFeatureExtractor) をインクリメンタルに計算
- `speakerCache.js` — Arrival-Order Speaker Cache + FIFO (Nemotron3DiarizationSpeakerCache)
- `session.js` — チャンクループ (Nemotron3DiarizationForAudioFrameClassification.forward)。`push(samples)` で確定したフレームの話者確率を返し、`finish()` で残りを処理する

モデルは初回にダウンロードし、Cache Storage / ブラウザキャッシュに保存する。

## ストリーミング処理

ファイルもマイクも同じ処理 (`diarizeStream` / `transcribeStream`) を通る。ファイルは最初から音声が揃っているだけ。
「マイクで録音を開始」でマイクから直接処理し、停止すると残りを処理して録音を WAV として再生できる。

- `capture.worklet.js` — AudioWorklet でマイク音声を 16 kHz mono に間引き、100 ms ずつ送る (AudioContext 自体はデバイスのレートのまま)
- 話者分離は最初からストリーミング API なので、届いた分を最大 10 秒ずつ worker に `push` する。マイクでレイテンシー設定が「オフライン」のときは `low_latency` (1.04 s) に切り替える
  - 低遅延設定は 0.72 s ごとに推論するが、WASM では 1 回 0.7〜1.2 s かかり追いつかない。マイク入力では推論中にたまったチャンクを 1 回の推論でまとめて処理する (`catchUp`) ので、遅れは積み上がらない (150 秒の録音で WASM 最大 2.3 s、WebGPU 最大 1.0 s)
- 文字起こしは未確定区間の先頭から最大 30 秒を Whisper に渡し、確定した単語の終わりから次の窓を始める。窓の終わりにかかる最後の単語は切れている可能性があるので次の窓でかけ直す
  - 録音中は 1〜2 秒ごとに区切る。単語の時刻は ±0.2 秒ほどずれるので、単語の終わりで区切ると次の単語の頭が欠ける。そこで録音中は、話者分離で 0.2 秒以上誰も話していない区間の中で区切り、区切れる無音がなければ確定を待つ (強制確定時だけ単語の終わりで区切る)。ファイルは区切りが窓ごとに 1 回なので単語の終わりで区切る (無音区間で区切ると窓の進みが小さくなり、文字起こしが 3 割ほど遅くなった)
  - Whisper が同じ句を延々と繰り返す幻覚を起こした窓は、繰り返しの手前までを確定し、続きを別の位置から始まる窓でかけ直す
  - Whisper は無音からも「ご視聴ありがとうございました」のような文章を作るので、話者分離の確率を VAD として使う。誰も話していない区間は飛ばし、無音区間に落ちた単語は捨てる
  - Whisper は入力を 30 秒にゼロ埋めするため、窓の外の時刻を返すことがある。単語の時刻は窓の中に収める
  - ファイル (と録音停止後) は窓全体の話者分離を待ってから 1 回だけかける
  - 録音中は低遅延のため 1 秒ごとに同じ区間をかけ直し、連続する 2 回の結果で先頭が一致した単語を確定する (LocalAgreement)。未確定区間が 20 秒を超えたら最後の 1 語以外を強制的に確定する。文字起こしは話者分離の遅れの分だけ遅れる
- 話者は確定済み + 未確定の単語に対して `toTurns(words, probs)` をその都度呼んで割り当てる

## ライセンス

このリポジトリのコードは [MIT](LICENSE)。`src/diarization/` は Hugging Face transformers (Apache-2.0) の Nemotron3Diarization 実装を移植したもの。
モデルは同梱せず実行時に取得する: Nemotron-3-Diarization は NVIDIA の [OpenMDW-1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization)、Whisper は MIT。
