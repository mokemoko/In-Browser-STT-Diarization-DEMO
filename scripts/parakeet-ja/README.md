# parakeet-ja の ONNX 変換

[yotarokubo/parakeet-hf-ctc-0.6b-ja](https://huggingface.co/yotarokubo/parakeet-hf-ctc-0.6b-ja)（[nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) の CTC ヘッドを Transformers 形式にしたもの）を、
[onnx-community/parakeet-ctc-0.6b-ONNX](https://huggingface.co/onnx-community/parakeet-ctc-0.6b-ONNX) と同じ入出力の ONNX にして
[mokemoko/parakeet-ctc-0.6b-ja-ONNX](https://huggingface.co/mokemoko/parakeet-ctc-0.6b-ja-ONNX) に置いている。

```sh
uv venv -p 3.12 .venv && uv pip install -p .venv torch transformers onnx onnxruntime onnxscript librosa huggingface_hub
.venv/bin/hf download yotarokubo/parakeet-hf-ctc-0.6b-ja --exclude 'converter/*' --local-dir hf
.venv/bin/python export.py hf out/model.onnx                # fp32 (重みはテンソルごとのファイルに分かれる)
.venv/bin/python quantize.py out/model.onnx model/onnx      # model / model_int8 (WASM 用) / model_q4 (WebGPU 用)
.venv/bin/python verify.py hf model/onnx sample.wav         # PyTorch との一致を確認
```

`model/` には `hf/` の `tokenizer.json` / `tokenizer_config.json` と、次の 2 つを置く。

- `config.json` — `hf/config.json` の `pad_token_id` を 3072 (CTC の blank = `<pad>`) に直し、`"transformers.js_config": {"use_external_data_format": true}` を足す
- `preprocessor_config.json` — `hf/processor_config.json` の `feature_extractor` に `"processor_class": "ParakeetProcessor"` を足す
