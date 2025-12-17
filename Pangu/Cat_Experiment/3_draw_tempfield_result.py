# 3_draw_tempfield_result.py
# 绘制全球地面温度场与稀疏风羽图

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 配置参数
# ==========================================
# 要绘制的时次列表 (小时): [0, 6, 12, 100, 1206] 等
# 0 表示初始场 (Input), >0 表示预报场 (Output)
SPECIFIED_HOURS = [0, 6, 12, 24, 48, 72]
# ==========================================

def draw_temperature_and_wind(npy_file_path, output_path=None):
    """
    绘制温度场和风场
    
    Args:
        npy_file_path: 输入的 surface .npy 文件路径
        output_path: 输出图片路径，如果为None则自动生成
    """
    if not os.path.exists(npy_file_path):
        print(f"[ERROR] File not found: {npy_file_path}")
        return

    print(f"[INFO] Loading data from: {npy_file_path}")
    try:
        # 加载数据 [4, 720, 1440] -> [mslet, u10, v10, t2m]
        data = np.load(npy_file_path)
        u10 = data[1]
        v10 = data[2]
        t2m = data[3]
        
        # 转换为摄氏度
        t2m_c = t2m - 273.15
        
        # 经纬度网格
        # 根据数据形状动态生成网格 (721x1440)
        nlat = t2m.shape[0]
        nlon = t2m.shape[1]
        print(f"[INFO] Data shape: {nlat}x{nlon}")
        
        lat = np.linspace(90, -90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        lons, lats = np.meshgrid(lon, lat)
        
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return

    # 计算数据统计量
    t2m_c = t2m - 273.15
    max_temp = np.max(t2m_c)
    min_temp = np.min(t2m_c)
    print(f"[INFO] Temperature min: {min_temp:.2f}°C, max: {max_temp:.2f}°C")
    
    # 设置绘图
    fig = plt.figure(figsize=(10, 7))
    # 修改投影中心为 180 度，使地图显示范围为 0-360 度 (即 0-180-0 视图)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    # 添加地理要素 (仅海陆边界，无国界)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, color='black')
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.3)
    
    # 绘制温度场 (填色图)
    print("[INFO] Plotting temperature field...")
    print("[INFO] This may take a moment for large datasets...")
    
    # 固定色条范围：±200°C间隔
    vmin = -200
    vmax = 200
    
    # pcolormesh 比 contourf 快得多
    # 不进行投影转换，直接按 0-360 经度绘制
    cf = ax.pcolormesh(lons, lats, t2m_c, 
                       cmap='RdYlBu_r', 
                       vmin=vmin, vmax=vmax,
                       transform=ccrs.PlateCarree(),
                       shading='auto')
    
    print("[INFO] Temperature field plotted successfully")
    
    # 添加色条（参考 draw_pangu_results.py 的风格，在右侧）
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.02, bbox.height])
    cbar = plt.colorbar(cf, cax=cax, orientation='vertical',
                       label='Temperature (°C)', ticks=np.arange(-200, 201, 100))
    
    # 绘制风羽 (稀疏)
    print("[INFO] Plotting sparse wind barbs...")
    # 稀疏采样：每隔 40 个点采一个 (1440/40 = 36 个箭头经向)
    step = 90 
    skip = (slice(None, None, step), slice(None, None, step))
    
    ax.barbs(lons[skip], lats[skip], u10[skip], v10[skip], 
             length=5, pivot='middle', linewidth=0.5, color='black', transform=ccrs.PlateCarree())
    
    # 提取文件信息用于标题
    filename = os.path.basename(npy_file_path)
    # 文件名格式：
    # Output: output_surface_20251217+00006h_CAT.npy
    # Input:  input_surface_20251217_CAT.npy (假设)
    try:
        parts = filename.split('_')
        if len(parts) >= 3:
            # 去除文件后缀 .npy 带来的干扰
            time_part = parts[2].replace('.npy', '')
            
            if '+' in time_part:
                # 格式如: 20251217+00006h
                init_time = time_part.split('+')[0]
                forecast_hour = time_part.split('+')[1] # 00006h
            else:
                # 格式如: 20251217 (Input文件)
                init_time = time_part
                forecast_hour = "0h"
        else:
            init_time = "Unknown"
            forecast_hour = ""
    except:
        init_time = "Unknown"
        forecast_hour = ""
    
    # 标题格式参考 draw_pangu_results.py
    # 在标题中增加显示 forecast_hour 以区分 0h 和其他时次
    ax.set_title(
        f'Global Surface Temperature (°C) & 10m Wind (m/s)\nInit: {init_time} {forecast_hour} | Min: {min_temp:.1f}°C | Max: {max_temp:.1f}°C',
        loc='left', fontsize=13
    )
    ax.set_title(f'Pangu-CAT', loc='right', fontsize=13)
    
    # 保存图片
    if output_path is None:
        # 统一保存路径逻辑
        dir_name = os.path.dirname(npy_file_path)
        
        # 无论是 Input 还是 Output，都尝试映射到 Run-output-png
        if 'Output' in dir_name:
            output_dir = dir_name.replace('Output', 'Run-output-png')
        elif 'Input' in dir_name:
            output_dir = dir_name.replace('Input', 'Run-output-png')
        else:
            # 如果不在标准目录结构中，则保存在当前目录下的 png 文件夹
            output_dir = os.path.join(dir_name, 'png')
            
        os.makedirs(output_dir, exist_ok=True)
            
        # 构建输出文件名
        base_name = os.path.splitext(filename)[0]
        # 如果是 Input 文件且文件名中没有 +xxxxx h，手动加上 +00000h 以保持文件名格式统一
        if 'input' in base_name.lower() and '+' not in base_name:
            base_name += "+00000h"
            
        output_filename = base_name + '_temp_wind.png'
        output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Saving figure to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Done.")

def process_single_hour(target_hour, project_root):
    """处理单个时次的查找与绘图"""
    target_file = None
    
    # 根据 target_hour 设置决定搜索目录
    if target_hour == 0:
        # 0h -> Input 目录
        default_dir = os.path.join(project_root, 'Input/Pangu/Cat_Experiment')
        file_prefix = 'input_surface'
        print(f"\n[INFO] === Processing Hour {target_hour} (Input) ===")
    else:
        # >0h -> Output 目录
        default_dir = os.path.join(project_root, 'Output/Pangu/Cat_Experiment')
        file_prefix = 'output_surface'
        print(f"\n[INFO] === Processing Hour {target_hour} (Forecast) ===")
    
    if os.path.exists(default_dir):
        files = [f for f in os.listdir(default_dir) if f.startswith(file_prefix) and f.endswith('.npy')]
        
        if target_hour > 0:
            # 对预报场进行时次筛选
            hour_str = f"+{target_hour:05d}h"
            files = [f for f in files if hour_str in f]
            if not files:
                print(f"[WARN] No output files found for hour {target_hour} (pattern: {hour_str})")
                return
        
        if files:
            # 按修改时间排序，找最新的
            files.sort(key=lambda x: os.path.getmtime(os.path.join(default_dir, x)), reverse=True)
            target_file = os.path.join(default_dir, files[0])
            print(f"[INFO] Found target file: {target_file}")
            draw_temperature_and_wind(target_file)
        else:
            print(f"[WARN] No matching {file_prefix} files found for hour {target_hour}.")
            return
    else:
        print(f"[ERROR] Directory not found: {default_dir}")
        return

def main():
    # 自动定位项目根目录
    project_root = os.getcwd()
    if not os.path.exists(os.path.join(project_root, 'Output')):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current = script_dir
        for _ in range(4):
            if os.path.exists(os.path.join(current, 'Output')):
                project_root = current
                break
            current = os.path.dirname(current)
    
    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Target hours: {SPECIFIED_HOURS}")
    
    for hour in SPECIFIED_HOURS:
        process_single_hour(hour, project_root)

if __name__ == "__main__":
    main()
