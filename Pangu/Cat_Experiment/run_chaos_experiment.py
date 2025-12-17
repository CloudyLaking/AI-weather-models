import os
import sys
import numpy as np
import onnxruntime as ort
from datetime import datetime

from draw_pangu_results import PanguResultDrawer

class PanguChaosRunner:
    """Pangu 混沌实验运行程序"""
    
    def __init__(self, 
                 input_dir='Input/Pangu/Cat_Experiment',
                 output_dir='Output/Pangu/Cat_Experiment',
                 image_dir='Run-output-png/Pangu/Cat_Experiment',
                 model_dir='Models-weights/Pangu'):
        
        project_root = os.getcwd()
        if not os.path.exists(os.path.join(project_root, 'Input')):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            current = script_dir
            for _ in range(4):
                if os.path.exists(os.path.join(current, 'Input')):
                    project_root = current
                    break
                current = os.path.dirname(current)
        
        self.input_dir = os.path.join(project_root, input_dir)
        self.output_dir = os.path.join(project_root, output_dir)
        self.image_dir = os.path.join(project_root, image_dir)
        self.model_dir = os.path.join(project_root, model_dir)
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        self.drawer = PanguResultDrawer(output_dir=self.image_dir) if PanguResultDrawer else None
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
        """100步内全部保存，100步后每10步保存一次"""
        return step < 100 or (step - 100) % 10 == 0
    
    def load_input_data(self):
        """加载输入数据"""
        upper_path = os.path.join(self.input_dir, 'input_upper.npy')
        surface_path = os.path.join(self.input_dir, 'input_surface.npy')
        
        if not os.path.exists(upper_path) or not os.path.exists(surface_path):
            print(f"[ERROR] Input files not found")
            return None, None
        
        try:
            input_upper = np.load(upper_path).astype(np.float32)
            input_surface = np.load(surface_path).astype(np.float32)
            print(f"[OK] Input data loaded (upper: {input_upper.shape}, surface: {input_surface.shape})")
            return input_upper, input_surface
        except Exception as e:
            print(f"[ERROR] Failed to load input: {e}")
            return None, None
    
    def run_chaos_experiment(self, total_steps=100000, init_datetime_str='20251217'):
        """运行混沌实验"""
        if self.ort_session is None:
            print("[ERROR] ONNX Session not initialized")
            return []
        
        current_upper, current_surface = self.load_input_data()
        if current_upper is None or current_surface is None:
            print("[ERROR] Failed to load input data")
            return []
        
        output_files = []
        print(f"\n[START] Running {total_steps} steps\n")
        
        for step in range(total_steps):
            hours = self.model_type * (step + 1)
            
            if (step + 1) % 1000 == 0:
                print(f"[RUN] Step {step+1}/{total_steps}: +{hours:05d}h", flush=True)
            
            try:
                output_upper, output_surface = self.ort_session.run(
                    None,
                    {'input': current_upper, 'input_surface': current_surface}
                )
            except Exception as e:
                print(f"[ERROR] Inference failed at step {step+1}: {e}")
                return output_files
            
            if self.should_save(step):
                try:
                    surface_filename = f'output_surface_{init_datetime_str}+{hours:05d}h_CAT.npy'
                    
                    np.save(os.path.join(self.output_dir, surface_filename), output_surface)
                    
                    output_files.append(os.path.join(self.output_dir, surface_filename))
                    
                    if self.drawer:
                        try:
                            data_dict = {
                                'mslp': output_surface[0] / 100,
                                'u10': output_surface[1],
                                'v10': output_surface[2],
                                't2m': output_surface[3],
                            }
                            self.drawer.draw_mslp_and_wind(
                                data_dict, init_datetime_str, hours,
                                data_source='CAT',
                                lon_range=None, lat_range=None
                            )
                        except Exception as e:
                            print(f"[WARN] Plotting failed: {e}")
                    
                    print(f"[SAVE] Step {step+1}: +{hours:05d}h saved")
                    
                except Exception as e:
                    print(f"[ERROR] Save failed at step {step+1}: {e}")
                    return output_files
            
            current_upper = output_upper.astype(np.float32)
            current_surface = output_surface.astype(np.float32)
        
        print(f"\n[OK] Completed! Saved {len(output_files)} files to {self.output_dir}")
        return output_files


def main():
    """主程序"""
    print("\n" + "="*70)
    print("PANGU CHAOS EXPERIMENT - 100000 STEPS")
    print("="*70)
    print(f"[START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    runner = PanguChaosRunner()
    
    start_time = datetime.now()
    output_files = runner.run_chaos_experiment(total_steps=100000, init_datetime_str='20251217')
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*70)
    print(f"[OK] Completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[TIME] Duration: {duration:.1f}s ({duration/3600:.2f}h)")
    print(f"[FILES] Saved {len(output_files)} files")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()