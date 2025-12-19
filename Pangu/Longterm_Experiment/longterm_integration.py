# AI-weather-models\Pangu\Longterm_Experiment\longterm_integration.py
# 长期自由积分实验：从1950/1990年开始积分到2020年

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import onnxruntime as ort
from scipy.ndimage import zoom

# 添加父目录到路径以导入模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'Main'))

try:
    from get_data_pangu import DataDownloader
    from draw_pangu_results import PanguResultDrawer
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    sys.exit(1)

# ==========================================
# 配置参数
# ==========================================
START_YEAR = 1990  # 初始年份：1950 或 1990
START_MONTH = 7    # 起报月份（例如 1990 年从 7 月 1 日开始）
START_DAY = 1      # 起报日
END_YEAR = 2020    # 结束年份
MODEL_TYPE = 6     # 使用6h模型（每天积分4次，仅00h保存）
# ==========================================

def find_project_root():
    """定位项目根目录"""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(current, 'Models-weights')):
            return current
        current = os.path.dirname(current)
    return os.getcwd()

def setup_directories(project_root, start_year):
    """创建目录结构 - 使用autodl-tmp数据盘"""
    exp_name = f'Longterm_{start_year}'
    
    # 检查是否有autodl-tmp目录（数据盘）
    data_disk = os.path.join(project_root, 'autodl-tmp')
    if os.path.exists(data_disk):
        print(f"[INFO] Using data disk: {data_disk}")
        base_output = data_disk
        base_input = data_disk
    else:
        print(f"[WARN] Data disk not found, using project root")
        base_output = project_root
        base_input = project_root
    
    dirs = {
        'input': os.path.join(base_input, 'Input', 'Pangu', exp_name),
        'raw_input': os.path.join(base_input, 'Input', 'Pangu_raw', exp_name),
        'output_daily': os.path.join(base_output, 'Output', 'Pangu', exp_name, 'Daily'),
        'output_full': os.path.join(base_output, 'Output', 'Pangu', exp_name, 'Full'),
        'images': os.path.join(base_output, 'Run-output-png', 'Pangu', exp_name, 'ERA5'),  # 修改：图片保存在ERA5子目录
        'model': os.path.join(project_root, 'Models-weights', 'Pangu')
    }
    
    for key, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory ready: {path}")
    
    return dirs

def coarsen_to_3deg(data, nlat=721, nlon=1440):
    """
    将数据从0.25度粗化到3度
    原始: 721x1440 (0.25度，包含南北极)
    目标: 61x120 (3度)
    """
    target_lat = 61  # 180/3 + 1 = 61 (包含南北极)
    target_lon = 120  # 360/3 = 120
    
    if data.ndim == 2:
        # 单层数据 [lat, lon]
        zoom_factor = (target_lat / nlat, target_lon / nlon)
        return zoom(data, zoom_factor, order=1)
    elif data.ndim == 3:
        # 多层数据 [level, lat, lon]
        result = []
        for i in range(data.shape[0]):
            zoom_factor = (target_lat / nlat, target_lon / nlon)
            result.append(zoom(data[i], zoom_factor, order=1))
        return np.array(result)
    else:
        raise ValueError(f"Unexpected data dimension: {data.ndim}")

def extract_pressure_levels(upper_data, levels=[850, 500, 200]):
    """
    从高空数据中提取特定气压层
    upper_data shape: [5, 13, 721, 1440]
    levels: 要提取的气压层索引
    
    气压层顺序（从高到低）:
    [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000] hPa
    """
    pressure_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    
    indices = []
    for level in levels:
        if level in pressure_levels:
            indices.append(pressure_levels.index(level))
        else:
            print(f"[WARN] Pressure level {level} not found, skipping")
    
    if not indices:
        raise ValueError("No valid pressure levels found")
    
    # upper_data: [5, 13, 721, 1440] -> [var, level, lat, lon]
    # 提取特定层
    extracted = upper_data[:, indices, :, :]  # [5, len(indices), 721, 1440]
    
    return extracted

def save_daily_coarse(surface_data, upper_data, output_path, current_date, forecast_hours=None):
    """
    保存每日粗化数据
    surface: [4, 721, 1440] -> 粗化到 [4, 61, 120]
    upper: [5, 13, 721, 1440] -> 提取850/500/200 -> [5, 3, 61, 120]
    """
    # 粗化地面场
    surface_coarse = coarsen_to_3deg(surface_data)
    
    # 提取并粗化高空场
    upper_extracted = extract_pressure_levels(upper_data, [850, 500, 200])
    upper_coarse = np.zeros((upper_extracted.shape[0], upper_extracted.shape[1], 61, 120))
    for var_idx in range(upper_extracted.shape[0]):
        for level_idx in range(upper_extracted.shape[1]):
            upper_coarse[var_idx, level_idx] = coarsen_to_3deg(upper_extracted[var_idx, level_idx])
    
    # 保存为两个独立的npy文件
    date_str = current_date.strftime('%Y%m%d')
    if forecast_hours is not None:
        # 新格式：包含预报时次
        surface_filename = f"daily_surface_{date_str}_+{forecast_hours:04d}h.npy"
        upper_filename = f"daily_upper_{date_str}_+{forecast_hours:04d}h.npy"
    else:
        # 旧格式：向后兼容
        surface_filename = f"daily_surface_{date_str}.npy"
        upper_filename = f"daily_upper_{date_str}.npy"
    
    surface_filepath = os.path.join(output_path, surface_filename)
    upper_filepath = os.path.join(output_path, upper_filename)
    
    np.save(surface_filepath, surface_coarse)
    np.save(upper_filepath, upper_coarse)
    
    return surface_filepath, upper_filepath

def save_full_data(surface_data, upper_data, output_path, current_date, forecast_hours=None):
    """保存完整数据"""
    date_str = current_date.strftime('%Y%m%d')
    if forecast_hours is not None:
        # 新格式：包含预报时次
        surface_filename = f"full_surface_{date_str}_+{forecast_hours:04d}h.npy"
        upper_filename = f"full_upper_{date_str}_+{forecast_hours:04d}h.npy"
    else:
        # 旧格式：向后兼容
        surface_filename = f"full_surface_{date_str}.npy"
        upper_filename = f"full_upper_{date_str}.npy"
    
    surface_filepath = os.path.join(output_path, surface_filename)
    upper_filepath = os.path.join(output_path, upper_filename)
    
    np.save(surface_filepath, surface_data)
    np.save(upper_filepath, upper_data)
    
    return surface_filepath, upper_filepath

def should_save_full(current_date):
    """判断是否应该保存完整数据（每年1月1日和7月1日）"""
    return (current_date.month == 1 and current_date.day == 1) or \
           (current_date.month == 7 and current_date.day == 1)

def find_last_saved_date(output_daily_dir, start_date):
    """查找最后保存的日期，兼容新旧格式"""
    if not os.path.exists(output_daily_dir):
        return None
    
    files = os.listdir(output_daily_dir)
    surface_files = [f for f in files if f.startswith('daily_surface_')]
    
    if not surface_files:
        return None
    
    last_date = None
    for filename in surface_files:
        # 提取日期：daily_surface_YYYYMMDD.npy 或 daily_surface_YYYYMMDD_+XXXXh.npy
        date_str = filename.replace('daily_surface_', '').split('_')[0].replace('.npy', '')
        try:
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if last_date is None or file_date > last_date:
                last_date = file_date
        except:
            continue
    
    return last_date

def find_last_saved_full_date(output_full_dir, start_date, end_date):
    """查找最后保存的完整数据日期"""
    if not os.path.exists(output_full_dir):
        return None
    
    files = os.listdir(output_full_dir)
    surface_files = [f for f in files if f.startswith('full_surface_')]
    
    if not surface_files:
        return None
    
    last_date = None
    for filename in surface_files:
        # 提取日期：full_surface_YYYYMMDD.npy 或 full_surface_YYYYMMDD_+XXXXh.npy
        date_str = filename.replace('full_surface_', '').split('_')[0].replace('.npy', '')
        try:
            file_date = datetime.strptime(date_str, '%Y%m%d')
            if (last_date is None or file_date > last_date) and file_date <= end_date:
                last_date = file_date
        except:
            continue
    
    return last_date

def run_longterm_integration(start_year, end_year=2020, start_month=1, start_day=1):
    """执行长期积分"""
    
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")
    
    # 设置目录
    dirs = setup_directories(project_root, start_year)
    
    # 初始化时间（支持自定义起报月日）
    current_date = datetime(start_year, start_month, start_day, 0)
    end_date = datetime(end_year, 12, 31, 0)
    
    print(f"\n{'='*70}")
    print(f"LONG-TERM INTEGRATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Start date: {current_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"End date:   {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Duration:   {(end_date - current_date).days} days")
    print(f"Model type: {MODEL_TYPE}h")
    print(f"{'='*70}\n")
    
    # Step 1: 获取初始条件
    print(f"[STEP 1] Acquiring initial conditions from ERA5...")
    init_datetime_str = current_date.strftime('%Y%m%d%H')
    
    # 初始化绘图器（提前初始化）
    drawer = PanguResultDrawer(output_dir=dirs['images'])
    
    try:
        downloader = DataDownloader(
            output_dir=dirs['input'],
            input_dir=dirs['input'],
            raw_input_dir=dirs['raw_input']
        )
        
        success = downloader.get_era5_data(init_datetime_str)
        
        if not success:
            print(f"[ERROR] Failed to acquire initial data")
            return
        
        # 加载初始数据
        upper_file = os.path.join(dirs['input'], 'input_upper.npy')
        surface_file = os.path.join(dirs['input'], 'input_surface.npy')
        
        if not os.path.exists(upper_file) or not os.path.exists(surface_file):
            print(f"[ERROR] Initial data files not found")
            return
        
        input_upper = np.load(upper_file).astype(np.float32)
        input_surface = np.load(surface_file).astype(np.float32)
        
        print(f"[OK] Initial conditions loaded")
        print(f"  Upper shape: {input_upper.shape}")
        print(f"  Surface shape: {input_surface.shape}")
        
        # 保存初始场数据
        try:
            # 保存初始场的每日粗化数据
            initial_daily = save_daily_coarse(
                input_surface, input_upper,
                dirs['output_daily'], current_date
            )
            
            # 保存初始场的完整数据
            initial_full = save_full_data(
                input_surface, input_upper,
                dirs['output_full'], current_date
            )
            print(f"[SAVE] {current_date.strftime('%Y-%m-%d')}: Initial field saved")
            
            # 绘制初始场图像
            try:
                data_dict = {
                    'mslp': input_surface[0] / 100,
                    'u10': input_surface[1],
                    'v10': input_surface[2],
                    't2m': input_surface[3],
                }
                
                init_str = current_date.strftime('%Y%m%d%H')
                forecast_hours = 0
                
                image_file = drawer.draw_mslp_and_wind(
                    data_dict, init_str, forecast_hours,
                    data_source='ERA5'
                )
                
                if image_file:
                    print(f"  [OK] Initial image saved: {os.path.basename(image_file)}")
            except Exception as e:
                print(f"  [WARN] Failed to draw initial image: {e}")
                
        except Exception as e:
            print(f"[ERROR] Failed to save initial field: {e}")
        
    except Exception as e:
        print(f"[ERROR] Failed to get initial conditions: {e}")
        return
    
    # Step 2: 加载模型
    print(f"\n[STEP 2] Loading Pangu-Weather {MODEL_TYPE}h model...")
    
    model_path = os.path.join(dirs['model'], f'pangu_weather_{MODEL_TYPE}.onnx')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return
    
    try:
        # 配置ONNX Session（参考run_chaos_experiment.py）
        options = ort.SessionOptions()
        options.enable_cpu_mem_arena = True
        options.enable_mem_pattern = True
        options.enable_mem_reuse = False
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 尝试使用CUDA加速
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
            print(f"[WARN] CUDA initialization failed, using CPU: {e}")
        
        ort_session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=providers
        )
        print(f"[OK] Model loaded: {model_path}")
        print(f"[INFO] Providers: {ort_session.get_providers()}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: 开始长期积分
    print(f"\n[STEP 3] Starting long-term integration...")
    
    # 检查是否有已保存的完整数据
    start_time = datetime(start_year, start_month, start_day, 0)
    last_saved = find_last_saved_full_date(dirs['output_full'], start_time, end_date)
    
    if last_saved and last_saved > start_time:
        print(f"[INFO] Found existing full data at {last_saved.strftime('%Y-%m-%d')}")
        print(f"[INFO] Resuming from {last_saved.strftime('%Y-%m-%d %H:%M')}")
        current_date = last_saved
        
        # 从完整数据加载最后的状态
        date_str = last_saved.strftime('%Y%m%d')
        
        # 尝试加载完整数据作为初始条件
        import glob
        full_surface_pattern = os.path.join(dirs['output_full'], f"full_surface_{date_str}_*.npy")
        full_upper_pattern = os.path.join(dirs['output_full'], f"full_upper_{date_str}_*.npy")
        
        surface_files = glob.glob(full_surface_pattern)
        upper_files = glob.glob(full_upper_pattern)
        
        if not surface_files:
            # 尝试旧格式
            surface_files = [os.path.join(dirs['output_full'], f"full_surface_{date_str}.npy")]
            upper_files = [os.path.join(dirs['output_full'], f"full_upper_{date_str}.npy")]
        
        if surface_files and upper_files and os.path.exists(surface_files[0]) and os.path.exists(upper_files[0]):
            input_surface = np.load(surface_files[0]).astype(np.float32)
            input_upper = np.load(upper_files[0]).astype(np.float32)
            print(f"[OK] Loaded saved state from {last_saved.strftime('%Y-%m-%d')}")
        else:
            print(f"[WARN] Full data files not found for {last_saved.strftime('%Y-%m-%d')}")
            print(f"[INFO] Starting from beginning")
            current_date = start_time
    else:
        print(f"[INFO] No existing full data found, starting from beginning")
        current_date = start_time
    
    step_count = 0
    day_count = int((current_date - start_time).days)
    
    while current_date <= end_date:
        step_count += 1
        
        # 模型预报
        try:
            output_upper, output_surface = ort_session.run(
                None,
                {
                    'input': input_upper,
                    'input_surface': input_surface
                }
            )
        except Exception as e:
            print(f"[ERROR] Model run failed at {current_date}: {e}")
            break
        
        # 更新当前时间
        current_date += timedelta(hours=MODEL_TYPE)
        
        # 计算预报时次
        forecast_hours = int((current_date - start_time).total_seconds() / 3600)
        
        # 显示进度（仅在每天00h）
        if current_date.hour == 0:
            day_count += 1
            print(f"[RUN] {current_date.strftime('%Y-%m-%d %H:%M')} (+{forecast_hours}h): Day {day_count}")
        
        # 保存数据（仅在每天00h）
        if current_date.hour == 0:
            # 每日粗化数据（始终保存，包含预报时次）
            try:
                daily_file = save_daily_coarse(
                    output_surface, output_upper,
                    dirs['output_daily'], current_date, forecast_hours
                )
                
                # 是否保存完整数据
                if should_save_full(current_date):
                    full_file = save_full_data(
                        output_surface, output_upper,
                        dirs['output_full'], current_date, forecast_hours
                    )
                    print(f"[SAVE] {current_date.strftime('%Y-%m-%d')} (+{forecast_hours}h): Full data saved")
                    
                    # 绘制图像
                    try:
                        data_dict = {
                            'mslp': output_surface[0] / 100,
                            'u10': output_surface[1],
                            'v10': output_surface[2],
                            't2m': output_surface[3],
                        }
                        
                        init_str = start_time.strftime('%Y%m%d%H')
                        
                        image_file = drawer.draw_mslp_and_wind(
                            data_dict, init_str, forecast_hours,
                            data_source='Pangu'
                        )
                        
                        if image_file:
                            print(f"  [OK] Image saved: {os.path.basename(image_file)}")
                    except Exception as e:
                        print(f"  [WARN] Failed to draw image: {e}")
                
            except Exception as e:
                print(f"[ERROR] Failed to save data at {current_date}: {e}")
        
        # 将输出作为下一步的输入
        input_upper = output_upper.astype(np.float32)
        input_surface = output_surface.astype(np.float32)
    
    print(f"\n{'='*70}")
    print(f"INTEGRATION COMPLETED")
    print(f"{'='*70}")
    print(f"Total steps: {step_count}")
    print(f"Total days: {day_count}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    print(f"\n[INFO] Starting long-term integration experiment")
    print(f"[INFO] Start year: {START_YEAR}")
    print(f"[INFO] End year: {END_YEAR}")
    print(f"[INFO] Model type: {MODEL_TYPE}h\n")
    
    # 直接使用配置参数运行
    run_longterm_integration(START_YEAR, END_YEAR, START_MONTH, START_DAY)
