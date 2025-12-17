# 混沌实验：Pangu 100000步演化程序
# 支持选择性保存：0-100步全保存，100-1000步每100步保存，1000步后每1000步保存

import os
import sys
import numpy as np
import onnxruntime as ort
from datetime import datetime

# 添加 Main 目录到路径以导入绘图模块
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), '../Main')))
try:
    from draw_pangu_results import PanguResultDrawer
except ImportError:
    print("[WARN] Could not import PanguResultDrawer, visualization will be disabled")
    PanguResultDrawer = None

class PanguChaosRunner:
    """Pangu 混沌实验运行程序"""
    
    def __init__(self, 
                 input_dir='Input/Pangu/Cat_Experiment',
                 output_dir='Output/Pangu/Cat_Experiment',
                 image_dir='Run-output-png/Pangu/Cat_Experiment',
                 model_dir='Models-weights/Pangu'):
        """
        初始化
        注意：默认路径假设当前工作目录为项目根目录。
        如果不是，程序会尝试自动寻找项目根目录。
        """
        # 尝试定位项目根目录 (包含 Input 文件夹的目录)
        project_root = os.getcwd()
        # 如果当前目录下没有 Input，尝试向上查找
        if not os.path.exists(os.path.join(project_root, 'Input')):
            # 尝试基于脚本位置查找
            script_dir = os.path.dirname(os.path.abspath(__file__))
            current = script_dir
            for _ in range(4):
                if os.path.exists(os.path.join(current, 'Input')):
                    project_root = current
                    break
                current = os.path.dirname(current)
        
        print(f"[INFO] Project root detected as: {project_root}")
        
        # 转换路径为绝对路径
        self.input_dir = os.path.join(project_root, input_dir)
        self.output_dir = os.path.join(project_root, output_dir)
        self.image_dir = os.path.join(project_root, image_dir)
        self.model_dir = os.path.join(project_root, model_dir)
        
        print(f"[INFO] Paths configured:")
        print(f"  Input:  {self.input_dir}")
        print(f"  Output: {self.output_dir}")
        print(f"  Images: {self.image_dir}")
        print(f"  Model:  {self.model_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 初始化绘图器
        if PanguResultDrawer:
            self.drawer = PanguResultDrawer(output_dir=self.image_dir)
        else:
            self.drawer = None
        
        # 初始化ONNX会话 (6h模型)
        self.ort_session = self._init_onnx_session(6)
        self.model_type = 6
    
    def _init_onnx_session(self, model_type):
        """初始化ONNX运行时"""
        model_path = os.path.join(self.model_dir, f'pangu_weather_{model_type}.onnx')
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None
        
        print(f"[LOAD] Loading ONNX model: {model_path}...")
        
        try:
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = True
            options.enable_mem_pattern = True
            options.enable_mem_reuse = False
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            providers = ['CPUExecutionProvider']
            try:
                cuda_provider_options = {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
                print("[OK] Using CUDA acceleration")
            except Exception as e:
                print(f"[WARN] CUDA failed, using CPU: {e}")
            
            session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
            print(f"[OK] Model loaded, providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            print(f"[ERROR] ONNX initialization failed: {e}")
            return None
    
    def should_save(self, step):
        """
        判断是否在该步骤保存
        
        Args:
            step: 迭代步数 (0-based)
            
        Returns:
            bool: 是否保存
        """
        if step < 100:
            # 0-100步：全部保存
            return True
        elif step < 1000:
            # 100-1000步：每100步保存一次
            return (step - 100) % 100 == 0
        else:
            # 1000步后：每1000步保存一次
            return (step - 1000) % 1000 == 0
    
    def load_input_data(self):
        """加载输入数据"""
        upper_path = os.path.join(self.input_dir, 'input_upper.npy')
        surface_path = os.path.join(self.input_dir, 'input_surface.npy')
        
        if not os.path.exists(upper_path) or not os.path.exists(surface_path):
            print(f"[ERROR] Input files not found")
            print(f"  upper: {upper_path} (exists={os.path.exists(upper_path)})")
            print(f"  surface: {surface_path} (exists={os.path.exists(surface_path)})")
            return None, None
        
        try:
            input_upper = np.load(upper_path).astype(np.float32)
            input_surface = np.load(surface_path).astype(np.float32)
            print(f"[OK] Input data loaded")
            print(f"  upper shape: {input_upper.shape}")
            print(f"  surface shape: {input_surface.shape}")
            return input_upper, input_surface
        except Exception as e:
            print(f"[ERROR] Failed to load input: {e}")
            return None, None
    
    def run_chaos_experiment(self, total_steps=100000, init_datetime_str='20251217'):
        """
        运行混沌实验
        
        Args:
            total_steps: 总迭代步数
            init_datetime_str: 初始时间字符串
        """
        if self.ort_session is None:
            print("[ERROR] ONNX Session not initialized")
            return []
        
        # 加载初始数据
        current_upper, current_surface = self.load_input_data()
        if current_upper is None or current_surface is None:
            print("[ERROR] Failed to load input data")
            return []
        
        output_files = []
        print(f"\n[START] Starting Pangu chaos experiment")
        print(f"  Total steps: {total_steps}")
        print(f"  Model type: {self.model_type}h")
        print(f"  Init time: {init_datetime_str}\n")
        
        # 绘制初始场 (0h)
        if self.drawer:
            try:
                print(f"[PLOT] Plotting initial state (0h)...")
                data_dict = {
                    'mslp': current_surface[0] / 100,  # Pa -> hPa
                    'u10': current_surface[1],
                    'v10': current_surface[2],
                    't2m': current_surface[3],
                }
                self.drawer.draw_mslp_and_wind(
                    data_dict, init_datetime_str, 0,
                    data_source='CAT',
                    lon_range=None, lat_range=None
                )
                print(f"[OK] Initial state plotted")
            except Exception as e:
                print(f"[WARN] Initial plotting failed: {e}")
        
        for step in range(total_steps):
            # 计算时效
            hours = self.model_type * (step + 1)
            
            # 判断是否保存
            should_save = self.should_save(step)
            
            # 运行推理
            if (step + 1) % 100 == 0 or step < 10:
                print(f"[RUN] Step {step+1}/{total_steps}: +{hours:05d}h ...", end=' ', flush=True)
            
            try:
                output_upper, output_surface = self.ort_session.run(
                    None,
                    {'input': current_upper, 'input_surface': current_surface}
                )
            except Exception as e:
                print(f"failed: {e}")
                return output_files
            
            # 保存文件
            if should_save:
                try:
                    upper_filename = f'output_upper_{init_datetime_str}+{hours:05d}h_CAT.npy'
                    surface_filename = f'output_surface_{init_datetime_str}+{hours:05d}h_CAT.npy'
                    
                    upper_path = os.path.join(self.output_dir, upper_filename)
                    surface_path = os.path.join(self.output_dir, surface_filename)
                    
                    np.save(upper_path, output_upper)
                    np.save(surface_path, output_surface)
                    
                    output_files.append(surface_path)
                    
                    # 绘制图像
                    if self.drawer:
                        try:
                            data_dict = {
                                'mslp': output_surface[0] / 100,  # Pa -> hPa
                                'u10': output_surface[1],
                                'v10': output_surface[2],
                                't2m': output_surface[3],
                            }
                            # 使用特殊的 data_source 标记以便区分
                            self.drawer.draw_mslp_and_wind(
                                data_dict, init_datetime_str, hours,
                                data_source='CAT',
                                lon_range=None, lat_range=None
                            )
                        except Exception as e:
                            print(f"[WARN] Plotting failed: {e}")
                    
                    if (step + 1) % 100 == 0 or step < 10:
                        print(f"saved & plotted")
                    
                except Exception as e:
                    print(f"save failed: {e}")
                    return output_files
            else:
                if (step + 1) % 100 == 0:
                    print(f"skipped")
            
            # 准备下一步输入
            current_upper = output_upper.astype(np.float32)
            current_surface = output_surface.astype(np.float32)
            
            # 进度显示
            if (step + 1) % 1000 == 0:
                print(f"[INFO] Progress: {step+1}/{total_steps} ({100*(step+1)/total_steps:.1f}%)")
        
        print(f"\n[OK] Chaos experiment completed!")
        print(f"  Total steps: {total_steps}")
        print(f"  Saved files: {len(output_files)}")
        print(f"  Output directory: {self.output_dir}")
        
        return output_files


def main():
    """主程序"""
    print("\n" + "="*70)
    print("PANGU CHAOS EXPERIMENT - 100000 STEPS")
    print("="*70)
    print(f"[START] Experiment started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    runner = PanguChaosRunner()
    
    experiment_start = datetime.now()
    output_files = runner.run_chaos_experiment(
        total_steps=100000,
        init_datetime_str='20251217'
    )
    experiment_end = datetime.now()
    duration = (experiment_end - experiment_start).total_seconds()
    
    print(f"\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"[OK] Experiment completed!")
    print(f"  Start time: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End time: {experiment_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: {duration:.1f} seconds ({duration/3600:.1f} hours)")
    print(f"  Output files: {len(output_files)}")
    print(f"  Output directory: {runner.output_dir}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
