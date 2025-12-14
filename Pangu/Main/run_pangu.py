# AI-weather-models\Pangu\Main\run-pangu.py
# Load and run Pangu weather forecast model using ONNX Runtime

import os
import numpy as np
import onnx
import onnxruntime as ort
from datetime import datetime
import sys

class PanguRunner:
    """Pangu weather forecast model runner"""
    
    def __init__(self, model_type=24, input_dir='../../../Input/pangu', 
                 output_dir='../../../Output/pangu', model_dir='../../../Models-weights/Pangu'):
        """
        Initialize Pangu runner
        
        Args:
            model_type: Model type, 6 for 6-hour forecast, 24 for 24-hour forecast
            input_dir: Input data directory
            output_dir: Output data directory
            model_dir: Model weights directory
        """
        self.model_type = model_type
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_dir = model_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize ONNX Runtime session
        self.ort_session = self._init_onnx_session()
    
    def _init_onnx_session(self):
        """Initialize ONNX Runtime session"""
        model_path = os.path.join(self.model_dir, f'pangu_weather_{self.model_type}.onnx')
        
        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            return None
        
        print(f"[LOAD] Loading ONNX model: {model_path}...")
        
        try:
            # Set ONNX Runtime options
            options = ort.SessionOptions()
            options.enable_cpu_mem_arena = True
            options.enable_mem_pattern = True
            options.enable_mem_reuse = False
            options.intra_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Try to use CUDA acceleration
            providers = ['CPUExecutionProvider']
            try:
                cuda_provider_options = {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
                print("[OK] Attempting to use CUDA acceleration")
            except Exception as e:
                print(f"[WARN] CUDA initialization failed, using CPU: {e}")
            
            session = ort.InferenceSession(model_path, sess_options=options, providers=providers)
            print(f"[OK] Model loaded successfully, using providers: {session.get_providers()}")
            return session
            
        except Exception as e:
            print(f"[ERROR] ONNX Runtime initialization failed: {e}")
            return None
    
    def load_input_data(self):
        """
        Load input data
        
        Returns:
            tuple: (input_upper, input_surface) or (None, None)
        """
        upper_path = os.path.join(self.input_dir, 'input_upper.npy')
        surface_path = os.path.join(self.input_dir, 'input_surface.npy')
        
        if not os.path.exists(upper_path) or not os.path.exists(surface_path):
            print(f"[ERROR] Input data files not found")
            print(f"  - upper: {upper_path} (exists={os.path.exists(upper_path)})")
            print(f"  - surface: {surface_path} (exists={os.path.exists(surface_path)})")
            return None, None
        
        try:
            input_upper = np.load(upper_path).astype(np.float32)
            input_surface = np.load(surface_path).astype(np.float32)
            print(f"[OK] Input data loaded successfully")
            print(f"  - upper shape: {input_upper.shape}")
            print(f"  - surface shape: {input_surface.shape}")
            return input_upper, input_surface
        except Exception as e:
            print(f"[ERROR] Failed to load input data: {e}")
            return None, None
    
    def run_forecast(self, run_times, datetime_str, data_source='GFS', skip_existing=True):
        """
        Run forecast
        
        Args:
            run_times: Number of forecast rounds (total forecast hours = run_times x model_type)
            datetime_str: Initialization time string in format 'YYYYMMDDHH'
            data_source: Data source label (for output filename)
            skip_existing: Whether to skip existing output files
            
        Returns:
            list: List of all output surface file paths
        """
        if self.ort_session is None:
            print("[ERROR] ONNX Session not initialized")
            return []
        
        # Load initial data
        current_upper, current_surface = self.load_input_data()
        if current_upper is None or current_surface is None:
            print("[ERROR] Unable to load input data, forecast aborted")
            return []
        
        output_files = []
        
        print(f"\n[START] Starting Pangu {self.model_type}h forecast")
        print(f"   Initialization time: {datetime_str}")
        print(f"   Forecast rounds: {run_times}")
        print(f"   Data source: {data_source}\n")
        
        for i in range(run_times):
            hour = self.model_type * i
            next_hour = self.model_type * (i + 1)
            
            output_upper_filename = f'output_upper_{datetime_str}+{next_hour:03d}h_{data_source}.npy'
            output_surface_filename = f'output_surface_{datetime_str}+{next_hour:03d}h_{data_source}.npy'
            
            output_upper_path = os.path.join(self.output_dir, output_upper_filename)
            output_surface_path = os.path.join(self.output_dir, output_surface_filename)
            
            # Check whether to skip existing files
            if skip_existing and os.path.exists(output_upper_path) and os.path.exists(output_surface_path):
                print(f"[SKIP] [{i+1}/{run_times}] +{next_hour:03d}h - File already exists, skipping")
                output_files.append(output_surface_path)
                
                # Load existing output as next iteration input
                current_upper = np.load(output_upper_path).astype(np.float32)
                current_surface = np.load(output_surface_path).astype(np.float32)
                continue
            
            # Run inference
            print(f"[RUN] [{i+1}/{run_times}] Running inference: +{hour:03d}h -> +{next_hour:03d}h ...", end=' ', flush=True)
            try:
                output_upper, output_surface = self.ort_session.run(
                    None, 
                    {'input': current_upper, 'input_surface': current_surface}
                )
                print("completed")
            except Exception as e:
                print(f"failed: {e}")
                return output_files
            
            # Save output data
            try:
                np.save(output_upper_path, output_upper)
                np.save(output_surface_path, output_surface)
                print(f"   [OK] Output saved: {output_surface_filename}")
                output_files.append(output_surface_path)
                
                # Prepare input for next iteration
                current_upper = output_upper.astype(np.float32)
                current_surface = output_surface.astype(np.float32)
                
            except Exception as e:
                print(f"   [ERROR] Save failed: {e}")
                return output_files
        
        print(f"\n[OK] Forecast completed! Generated {len(output_files)} time steps of output")
        return output_files
    
    def get_forecast_data(self, output_surface_path):
        """
        Read forecast result data
        
        Args:
            output_surface_path: Output surface layer file path
            
        Returns:
            dict: Dictionary containing forecast data
        """
        try:
            data = np.load(output_surface_path)
            return {
                'mslp': data[0] / 100,  # Convert to hPa
                'u10': data[1],
                'v10': data[2],
                't2m': data[3],
                'raw_data': data
            }
        except Exception as e:
            print(f"[ERROR] Failed to read forecast data: {e}")
            return None


def run_pangu_forecast(model_type=24, run_times=8, datetime_str='2025121200', 
                      data_source='GFS', skip_existing=True):
    """
    Convenience function: One-click forecast execution
    
    Args:
        model_type: Model type, 1/3/6/24
        run_times: Number of forecast rounds
        datetime_str: Initialization time
        data_source: Data source
        skip_existing: Whether to skip existing files
        
    Returns:
        list: Output file list
    """
    runner = PanguRunner(model_type=model_type)
    return runner.run_forecast(run_times, datetime_str, data_source, skip_existing)


if __name__ == '__main__':
    # Example usage
    runner = PanguRunner(model_type=24) # 1/3/6/24
    output_files = runner.run_forecast(
        run_times=8,   # Number of forecast rounds
        datetime_str='2025120100',   # Initialization time
        data_source='ERA5',  #GFS/ERA5/ECMWF
        skip_existing=True  
    )
    
    print(f"\nGenerated output files:")
    for f in output_files:
        print(f"  - {f}")
