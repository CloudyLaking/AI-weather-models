import os
import numpy as np
import onnx
import onnxruntime as ort

# 输入和输出数据的目录
input_data_dir = r'input'
output_data_dir = r'output'

print("Loading ONNX model...")
model = onnx.load(r'C:\Users\lyz13\OneDrive\Desktop\AI_models_research\Pangu-Weather-Release\weight\pangu_weather_6.onnx')

# 设置 onnxruntime 的行为
print("Setting ONNX Runtime options...")
options = ort.SessionOptions()
options.enable_cpu_mem_arena = True
options.enable_mem_pattern = True
options.enable_mem_reuse = False
# 增加线程数以加快推理速度，但会消耗更多内存
options.intra_op_num_threads = 1

# 设置 CUDA provider 的行为
print("Setting CUDA provider options...")
cuda_provider_options = {
    'arena_extend_strategy': 'kSameAsRequested',
    'gpu_mem_limit': 20 * 1024 * 1024 * 1024,  # 设置 GPU 内存限制，例如 2GB
    'cudnn_conv_algo_search': 'EXHAUSTIVE',  # 使用最优的 cuDNN 卷积算法
    'do_copy_in_default_stream': True,
}

# 初始化 ONNX Runtime 会话以使用 Pangu-Weather 模型
print("Initializing ONNX Runtime session...")
ort_session = ort.InferenceSession(
    r'C:\Users\lyz13\OneDrive\Desktop\AI_models_research\Pangu-Weather-Release\weight\pangu_weather_6.onnx',
    sess_options=options,
    providers=[('CUDAExecutionProvider', cuda_provider_options)]
)

# 加载高空 numpy 数组
print("Loading input data...")
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# 加载地表 numpy 数组
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

# 运行推理会话
print("Running inference...")
output, output_surface = ort_session.run(None, {'input': input, 'input_surface': input_surface})

# 保存结果
print("Saving results...")
np.save(os.path.join(output_data_dir, 'output_upper.npy'), output)
np.save(os.path.join(output_data_dir, 'output_surface.npy'), output_surface)

print('Done!')