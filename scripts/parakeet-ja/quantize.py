# 書き出した ONNX の重みを model*.onnx_data に一本化し、int8 (wasm 用) / q4 (webgpu 用) を作る
import sys, os, onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)


def save(m, name):
    onnx.save(m, f'{dst}/{name}.onnx', save_as_external_data=True, all_tensors_to_one_file=True,
              location=f'{name}.onnx_data', size_threshold=0)
    print(name, os.path.getsize(f'{dst}/{name}.onnx_data') // 2**20, 'MiB')


save(onnx.load(src), 'model')

quantize_dynamic(f'{dst}/model.onnx', f'{dst}/tmp_int8.onnx', weight_type=QuantType.QInt8,
                 op_types_to_quantize=['MatMul'], use_external_data_format=True)
save(onnx.load(f'{dst}/tmp_int8.onnx'), 'model_int8')

q = MatMulNBitsQuantizer(onnx.load(f'{dst}/model.onnx'), block_size=32, is_symmetric=True)
q.process()
save(q.model.model, 'model_q4')
