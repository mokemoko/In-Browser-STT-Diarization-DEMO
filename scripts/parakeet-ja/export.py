# yotarokubo/parakeet-hf-ctc-0.6b-ja を Transformers.js 形式 (onnx-community/parakeet-ctc-0.6b-ONNX と同じ I/O) に書き出す
import sys, torch
from transformers import ParakeetForCTC

src, out = sys.argv[1], sys.argv[2]
model = ParakeetForCTC.from_pretrained(src).eval()


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_features, attention_mask):
        return self.m(input_features=input_features, attention_mask=attention_mask).logits


T = 1000  # 10 秒分
x = torch.randn(1, T, 80)
mask = torch.ones(1, T, dtype=torch.long)
torch.onnx.export(
    Wrap(model), (x, mask), out,
    input_names=['input_features', 'attention_mask'], output_names=['logits'],
    dynamic_axes={'input_features': {0: 'batch_size', 1: 'sequence_length'},
                  'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                  'logits': {0: 'batch_size', 1: 'sequence_length'}},
    opset_version=18, dynamo=False, external_data=True,
)
print('exported', out)
