# PyTorch と ONNX (fp32 / int8 / q4) の出力を日本語音声で比較する
import sys, numpy as np, torch, librosa, onnxruntime as ort
from transformers import AutoProcessor, ParakeetForCTC

hf, onnx_dir, wav = sys.argv[1:4]
proc = AutoProcessor.from_pretrained(hf)
audio, _ = librosa.load(wav, sr=16000)
inp = proc(audio, sampling_rate=16000, return_tensors='pt')


def decode(ids):
    out, prev = [], None
    for i in ids:
        if i != prev and i != 3072:  # blank は語彙の末尾
            out.append(int(i))
        prev = i
    return proc.tokenizer.decode(out)


with torch.no_grad():
    ref = ParakeetForCTC.from_pretrained(hf).eval()(**inp).logits[0].numpy()
print('torch :', decode(ref.argmax(-1)))
feeds = {'input_features': inp['input_features'].numpy(), 'attention_mask': inp['attention_mask'].numpy().astype(np.int64)}
for name in ['model', 'model_int8', 'model_q4']:
    lg = ort.InferenceSession(f'{onnx_dir}/{name}.onnx').run(None, feeds)[0][0]
    print(f'{name:10s}:', decode(lg.argmax(-1)), '| max|diff|', float(np.abs(lg - ref).max()), '| argmax一致率', float((lg.argmax(-1) == ref.argmax(-1)).mean()))
