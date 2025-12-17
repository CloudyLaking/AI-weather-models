# 5_draw_time_series.py
# 绘制全球平均 2mT、10m风速、MSLP 的时间序列图

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# ==========================================
# 配置参数
# ==========================================
FIG_SIZE = (12, 5)
MAX_HOUR = 8466  # 最大预报时次
# 颜色配置
COLOR_T2M = "#FF0000"    # 温度颜色
COLOR_WIND = "#1E90FF"   # 风速颜色
COLOR_MSLP = "#409900"   # 气压颜色
# ==========================================

def find_project_root():
    """自动定位项目根目录"""
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.exists(os.path.join(current, 'Output')):
            return current
        current = os.path.dirname(current)
    return os.getcwd()

def find_all_files(project_root):
    """查找所有可用的 surface 文件（Input 和 Output）"""
    files_dict = {}
    
    # 1. 查找 Input 文件 (0h)
    input_dir = os.path.join(project_root, 'Input/Pangu/Cat_Experiment')
    if os.path.exists(input_dir):
        input_files = glob(os.path.join(input_dir, 'input_surface*.npy'))
        for f in input_files:
            files_dict[0] = f
            print(f"[INFO] Found 0h file: {os.path.basename(f)}")
    
    # 2. 查找 Output 文件 (>0h)
    output_dir = os.path.join(project_root, 'Output/Pangu/Cat_Experiment')
    if os.path.exists(output_dir):
        output_files = glob(os.path.join(output_dir, 'output_surface*.npy'))
        for f in output_files:
            # 从文件名提取小时数
            try:
                basename = os.path.basename(f)
                # 格式: output_surface_20251217+00024h_CAT.npy
                if '+' in basename:
                    hour_part = basename.split('+')[1].split('h')[0]
                    hour = int(hour_part)
                    if hour <= MAX_HOUR:
                        files_dict[hour] = f
                        print(f"[INFO] Found {hour}h file: {basename}")
            except Exception as e:
                print(f"[WARN] Failed to parse filename {basename}: {e}")
                continue
    
    return files_dict

def load_and_compute_global_mean(file_path):
    """加载数据并计算全球平均值"""
    try:
        data = np.load(file_path)
        # data shape: [4, 720, 1440] -> [mslp, u10, v10, t2m]
        mslp = data[0]
        u10 = data[1]
        v10 = data[2]
        t2m = data[3]
        
        # 转换单位
        t2m_c = t2m - 273.15  # K -> °C
        mslp_hpa = mslp / 100.0  # Pa -> hPa
        wind_speed = np.sqrt(u10**2 + v10**2)  # m/s
        
        # 全球平均（简单算术平均，未考虑纬度权重）
        mean_t2m = np.mean(t2m_c)
        mean_wind = np.mean(wind_speed)
        mean_mslp = np.mean(mslp_hpa)
        
        return mean_t2m, mean_wind, mean_mslp
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        return None, None, None

def draw_time_series():
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")
    
    # 查找所有文件
    files_dict = find_all_files(project_root)
    
    if not files_dict:
        print("[ERROR] No files found!")
        return
    
    # 提取时间序列数据
    hours = sorted(files_dict.keys())
    t2m_series = []
    wind_series = []
    mslp_series = []
    init_time_str = "Unknown"
    
    print(f"\n[INFO] Processing {len(hours)} time steps...")
    for hour in hours:
        file_path = files_dict[hour]
        mean_t2m, mean_wind, mean_mslp = load_and_compute_global_mean(file_path)
        
        if mean_t2m is not None:
            t2m_series.append(mean_t2m)
            wind_series.append(mean_wind)
            mslp_series.append(mean_mslp)
            
            # 提取初始时间（仅第一次）
            if init_time_str == "Unknown":
                try:
                    basename = os.path.basename(file_path)
                    parts = basename.split('_')
                    time_part = parts[2].replace('.npy', '')
                    init_time_str = time_part.split('+')[0] if '+' in time_part else time_part
                except:
                    pass
        else:
            print(f"[WARN] Skipping hour {hour} due to processing error")
            hours.remove(hour)
    
    if not t2m_series:
        print("[ERROR] No valid data to plot!")
        return
    
    print(f"[INFO] Successfully processed {len(hours)} time steps")
    print(f"[INFO] Hour range: {min(hours)} to {max(hours)}")
    
    # === 绘图 ===
    fig, ax1 = plt.subplots(figsize=FIG_SIZE)
    
    # 左Y轴：2m Temperature
    ax1.set_xlabel('Forecast Hour', fontsize=12)
    ax1.set_ylabel('Global Mean 2m Temperature (°C)', color=COLOR_T2M, fontsize=12)
    line1 = ax1.plot(hours, t2m_series, color=COLOR_T2M, linewidth=2, marker='o', 
                     markersize=4, label='2m Temperature')
    ax1.tick_params(axis='y', labelcolor=COLOR_T2M)
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 右Y轴1：10m Wind Speed
    ax2 = ax1.twinx()
    ax2.set_ylabel('10m Wind Speed (m/s)', color=COLOR_WIND, fontsize=12)
    line2 = ax2.plot(hours, wind_series, color=COLOR_WIND, linewidth=2, marker='s', 
                     markersize=4, label='10m Wind Speed')
    ax2.tick_params(axis='y', labelcolor=COLOR_WIND)
    ax2.set_ylim(0, 100)
    
    # 右Y轴2：MSLP
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('MSLP (hPa)', color=COLOR_MSLP, fontsize=12)
    line3 = ax3.plot(hours, mslp_series, color=COLOR_MSLP, linewidth=2, marker='^', 
                     markersize=4, label='MSLP')
    ax3.tick_params(axis='y', labelcolor=COLOR_MSLP)
    
    # 主标题（接续编号 g）
    fig.suptitle(f'(g) Pangu Global Mean Time Series', 
                 x=0.08, y=0.92, ha='left', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存
    output_dir = os.path.join(project_root, 'Run-output-png')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'Time_Series_T2m_Wind_MSLP_{init_time_str}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Saving time series figure to: {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Done.")

if __name__ == "__main__":
    draw_time_series()
