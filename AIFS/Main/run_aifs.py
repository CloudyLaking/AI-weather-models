# AI-weather-models\AIFS\Main\run_aifs.py
# 加载并运行 AIFS 天气预报模型

import os
import sys
import numpy as np
from datetime import datetime, timedelta

try:
    from anemoi.inference.runners.simple import SimpleRunner
    from anemoi.inference.outputs.printer import print_state
except ImportError:
    print("[ERROR] anemoi package not installed, run: pip install anemoi-inference anemoi-models")
    sys.exit(1)


class AIFSRunner:
    """AIFS 天气预报模型执行器"""
    
    # AIFS 模型检查点（从 Hugging Face，作为备选）
    CHECKPOINT_HF = {"huggingface": "ecmwf/aifs-single-1.0"}
    # 本地模型默认路径
    DEFAULT_LOCAL_MODEL = "Models-weights/AIFS/aifs-single-mse-1.0.ckpt"
    
    def __init__(self, device='cuda', num_chunks=None, 
                 model_path=None, output_dir='Output/AIFS'):
        """
        初始化 AIFS 执行器
        
        参数:
            device: 计算设备 'cuda' 或 'cpu'
            num_chunks: 模型映射器的分块数（降低内存占用）
            model_path: 本地模型文件路径，为 None 则尝试使用默认本地路径或 Hugging Face
            output_dir: 输出数据目录
        """
        # 如果是相对路径，则基于当前脚本所在目录转换为绝对路径
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
        
        self.output_dir = output_dir
        self.device = device
        self.num_chunks = num_chunks
        
        # 确定模型路径
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(script_dir, self.DEFAULT_LOCAL_MODEL))
        elif not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.normpath(os.path.join(script_dir, model_path))
        
        self.model_path = model_path
        self.use_local_model = os.path.exists(self.model_path)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 配置环境变量以减少内存占用
        if num_chunks is not None:
            os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = str(num_chunks)
        
        # 初始化 SimpleRunner
        self.runner = self._init_runner()
    
    def _init_runner(self):
        """初始化 AIFS SimpleRunner"""
        print(f"[LOAD] Initializing AIFS SimpleRunner...")
        print(f"  Device: {self.device}")
        
        if self.num_chunks is not None:
            print(f"  Num chunks: {self.num_chunks}")
        
        # 优先使用本地模型，否则从 Hugging Face 下载
        checkpoint = None
        if self.use_local_model:
            print(f"  Model (local): {self.model_path}")
            checkpoint = self.model_path
        else:
            print(f"  Model (Hugging Face): ecmwf/aifs-single-1.0")
            print(f"  Local model not found: {self.model_path}")
            print(f"  Will download from Hugging Face...")
            checkpoint = self.CHECKPOINT_HF
        
        try:
            runner = SimpleRunner(checkpoint, device=self.device)
            print(f"[OK] AIFS SimpleRunner initialized successfully")
            return runner
        except Exception as e:
            print(f"[ERROR] Failed to initialize AIFS SimpleRunner: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_forecast(self, input_state, lead_time=12, save_outputs=True, 
                    datetime_str=None, skip_existing=True):
        """
        执行 AIFS 预报
        
        参数:
            input_state: 输入状态字典（包含 date 和 fields）
            lead_time: 预报时效（小时）
            save_outputs: 是否保存输出数据
            datetime_str: 起报时间字符串 'YYYYMMDDHH'
            skip_existing: 是否跳过已存在的输出文件
            
        返回:
            list: 生成的输出文件路径列表
        """
        if self.runner is None:
            print("[ERROR] AIFS SimpleRunner not initialized")
            return []
        
        if input_state is None or 'date' not in input_state:
            print("[ERROR] Invalid input state")
            return []
        
        # 确定起报时间
        if datetime_str is None:
            datetime_str = input_state['date'].strftime('%Y%m%d%H')
        
        print(f"\n[START] Starting AIFS forecast")
        print(f"  Initialization time: {datetime_str}")
        print(f"  Lead time: {lead_time} hours")
        print(f"  Device: {self.device}\n")
        
        output_files = []
        forecast_states = []
        
        try:
            # 运行预报循环
            for state in self.runner.run(input_state=input_state, lead_time=lead_time):
                # 打印当前状态信息
                print_state(state)
                
                # 提取预报时效
                forecast_hour = int((state['date'] - input_state['date']).total_seconds() / 3600)
                
                # 保存输出
                if save_outputs:
                    output_file = self._save_forecast_state(
                        state, datetime_str, forecast_hour, skip_existing
                    )
                    if output_file:
                        output_files.append(output_file)
                
                forecast_states.append(state)
            
            print(f"\n[OK] Forecast completed! Generated {len(output_files)} forecast steps")
            return output_files
            
        except Exception as e:
            print(f"[ERROR] Forecast execution failed: {e}")
            import traceback
            traceback.print_exc()
            return output_files
    
    def _save_forecast_state(self, state, datetime_str, forecast_hour, skip_existing=True):
        """
        保存预报状态为 NPY 文件
        
        参数:
            state: 预报状态字典
            datetime_str: 起报时间字符串
            forecast_hour: 预报时效（小时）
            skip_existing: 是否跳过已存在的文件
            
        返回:
            str: 输出文件路径，或 None 如果保存失败
        """
        try:
            # 构建输出文件名
            output_filename = f'output_aifs_{datetime_str}+{forecast_hour:03d}h.npz'
            output_path = os.path.join(self.output_dir, output_filename)
            
            # 检查是否跳过
            if skip_existing and os.path.exists(output_path):
                print(f"[SKIP] Output file already exists: {output_filename}")
                return None
            
            # 提取字段数据
            fields = state.get('fields', {})
            
            # 保存为压缩的 NPZ 格式
            save_dict = {
                'date': state['date'].isoformat(),
                'latitudes': state.get('latitudes', np.array([])),
                'longitudes': state.get('longitudes', np.array([])),
            }
            
            # 添加所有字段
            for field_name, field_data in fields.items():
                if isinstance(field_data, np.ndarray):
                    save_dict[field_name] = field_data
            
            np.savez_compressed(output_path, **save_dict)
            print(f"[OK] Output saved: {output_filename}")
            return output_path
            
        except Exception as e:
            print(f"[ERROR] Failed to save forecast state: {e}")
            return None
    
    def load_forecast_state(self, output_path):
        """
        读取保存的预报状态
        
        参数:
            output_path: 输出文件路径
            
        返回:
            dict: 预报状态字典
        """
        try:
            data = np.load(output_path, allow_pickle=True)
            
            state = {
                'date': datetime.fromisoformat(str(data['date'])),
                'latitudes': data['latitudes'],
                'longitudes': data['longitudes'],
                'fields': {}
            }
            
            # 提取所有字段
            for key in data.files:
                if key not in ['date', 'latitudes', 'longitudes']:
                    state['fields'][key] = data[key]
            
            return state
            
        except Exception as e:
            print(f"[ERROR] Failed to load forecast state: {e}")
            return None


def run_aifs_forecast(input_state, lead_time=12, datetime_str=None, 
                     device='cuda', model_path=None, skip_existing=True):
    """
    便捷函数：一键执行 AIFS 预报
    
    参数:
        input_state: 输入状态字典（包含 date 和 fields）
        lead_time: 预报时效（小时）
        datetime_str: 起报时间字符串 'YYYYMMDDHH'
        device: 计算设备 'cuda' 或 'cpu'
        model_path: 本地模型文件路径（为 None 则使用默认或 Hugging Face）
        skip_existing: 是否跳过已存在的文件
        
    返回:
        list: 输出文件路径列表
    """
    runner = AIFSRunner(device=device, model_path=model_path)
    return runner.run_forecast(
        input_state,
        lead_time=lead_time,
        datetime_str=datetime_str,
        skip_existing=skip_existing
    )


if __name__ == '__main__':
    # 示例用法
    from get_data_aifs import get_aifs_data
    
    # 获取初始条件
    print("[INFO] Retrieving initial conditions...")
    input_state = get_aifs_data()
    
    if input_state is None:
        print("[ERROR] Failed to retrieve initial conditions")
        sys.exit(1)
    
    # 执行预报
    output_files = run_aifs_forecast(
        input_state,
        lead_time=12,  # 12小时预报
        device='cuda'
    )
    
    print(f"\nGenerated output files:")
    for f in output_files:
        print(f"  - {f}")
