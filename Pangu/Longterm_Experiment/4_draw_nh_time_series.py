# 4_draw_nh_time_series.py
# 绘制北半球10N-60N范围的时间序列图（2mT、10m风速、MSLP）
# 基于长期积分实验输出数据

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime, timedelta

# ==========================================
# 配置参数 - 用户可直接在这里修改
# ==========================================

# === 必要参数：请在这里设置 ===
PROJECT_ROOT = None  # 项目根目录。设为None则自动检测，或指定路径如: "/root/AI-weather-models"
START_YEAR = None    # 起始年份。设为None则自动检测，或指定如: 1950

# === 时间范围限制 ===
TIME_START = 1950    # 开始年份
TIME_END = 1965      # 结束年份

# === 绘图参数 ===
FIG_SIZE = (10, 5)  # 参考5_draw_time_series.py
MAX_HOUR = 999999  # 最大预报时次

# 北半球范围定义（0.25度格网）
# 全球: 721 lat x 1440 lon (0.25度)
# 纬度范围: -90到90度，经度范围: 0到360度
# 10N: 纬度索引 = (90 - 10) * 4 = 320
# 60N: 纬度索引 = (90 - 60) * 4 = 120
LAT_SOUTH = 10  # 南边界
LAT_NORTH = 60  # 北边界（北半球中高纬）
LAT_IDX_SOUTH = int((90 - LAT_SOUTH) * 4)  # 索引320
LAT_IDX_NORTH = int((90 - LAT_NORTH) * 4)  # 索引120

# 颜色配置
COLOR_T2M = "#FF0000"    # 温度颜色
COLOR_WIND = "#1E90FF"   # 风速颜色
COLOR_MSLP = "#409900"   # 气压颜色

# ==========================================

def find_project_root():
    """自动定位项目根目录"""
    current = os.path.dirname(os.path.abspath(__file__))
    
    # 方法1: 往上找 Models-weights 目录
    temp = current
    for _ in range(5):
        if os.path.exists(os.path.join(temp, 'Models-weights')):
            return temp
        temp = os.path.dirname(temp)
    
    # 方法2: 往上找包含 Output/Pangu 的目录
    temp = current
    for _ in range(5):
        if os.path.exists(os.path.join(temp, 'Output', 'Pangu')):
            print(f"[DEBUG] Found Output/Pangu at: {temp}")
            return temp
        temp = os.path.dirname(temp)
    
    # 方法3: 返回脚本所在的Longterm_Experiment父目录的父目录 (Pangu的父目录)
    # /root/AI-weather-models/Pangu/Longterm_Experiment -> /root/AI-weather-models
    pangu_dir = os.path.dirname(current)  # /root/AI-weather-models/Pangu
    project_dir = os.path.dirname(pangu_dir)  # /root/AI-weather-models
    
    print(f"[DEBUG] Using inferred project root: {project_dir}")
    return project_dir

def find_start_year_from_dirs(project_root):
    """从Output目录推断可用的实验年份"""
    
    # 根据longterm_integration.py的逻辑，优先检查autodl-tmp目录（数据盘）
    data_disk = os.path.join(project_root, 'autodl-tmp')
    if os.path.exists(data_disk):
        output_dir = os.path.join(data_disk, 'Output', 'Pangu')
        print(f"[DEBUG] Data disk found, looking in: {output_dir}")
    else:
        output_dir = os.path.join(project_root, 'Output', 'Pangu')
        print(f"[DEBUG] No data disk found, looking in: {output_dir}")
    
    print(f"[DEBUG] Looking for Longterm dirs in: {output_dir}")
    print(f"[DEBUG] Directory exists: {os.path.exists(output_dir)}")
    
    if not os.path.exists(output_dir):
        print(f"[WARN] Output directory not found: {output_dir}")
        return None
    
    # 查找所有Longterm_XXXX形式的目录
    try:
        subdirs = os.listdir(output_dir)
        print(f"[DEBUG] Found subdirs in Output/Pangu: {subdirs}")
    except Exception as e:
        print(f"[ERROR] Failed to list {output_dir}: {e}")
        return None
    
    for subdir in subdirs:
        print(f"[DEBUG] Checking subdir: {subdir}")
        if subdir.startswith('Longterm_'):
            try:
                start_year = int(subdir.split('_')[1])
                print(f"[DEBUG] Found start_year: {start_year}")
                return start_year
            except Exception as e:
                print(f"[DEBUG] Failed to parse {subdir}: {e}")
                continue
    
    print(f"[WARN] No Longterm_XXXX directories found")
    return None

def find_all_daily_files(project_root, start_year):
    """查找所有日均surface数据文件，支持新旧两种格式"""
    files_dict = {}  # {datetime.date: file_path}
    
    exp_name = f'Longterm_{start_year}'
    
    # 根据longterm_integration.py的逻辑，优先检查autodl-tmp目录（数据盘）
    data_disk = os.path.join(project_root, 'autodl-tmp')
    if os.path.exists(data_disk):
        base_output = data_disk
        print(f"[INFO] Data disk found at: {data_disk}")
    else:
        base_output = project_root
        print(f"[INFO] Using project root for data")
    
    # 查找Daily输出文件
    daily_dir = os.path.join(base_output, f'Output/Pangu/{exp_name}/Daily')
    
    if not os.path.exists(daily_dir):
        print(f"[WARN] Daily directory not found: {daily_dir}")
        return files_dict
    
    print(f"[INFO] Searching for daily files in: {daily_dir}")
    
    # 查找surface文件
    surface_files = glob(os.path.join(daily_dir, 'daily_surface_*.npy'))
    
    print(f"[INFO] Found {len(surface_files)} surface files")
    
    for f in surface_files:
        try:
            basename = os.path.basename(f)
            # 支持两种格式: 
            # - 旧格式: daily_surface_YYYYMMDD.npy
            # - 新格式: daily_surface_YYYYMMDD_+XXXXh.npy
            
            # 提取日期
            date_str = basename.replace('daily_surface_', '').split('_')[0].replace('.npy', '')
            file_date = datetime.strptime(date_str, '%Y%m%d').date()
            
            files_dict[file_date] = f
            print(f"[INFO] Found: {basename}")
            
        except Exception as e:
            print(f"[WARN] Failed to parse filename {basename}: {e}")
            continue
    
    return files_dict

def load_and_compute_regional_mean(file_path, lat_idx_north, lat_idx_south):
    """加载数据并计算北半球10N-60N范围的区域平均值"""
    try:
        data = np.load(file_path)
        # 原始数据 shape: [4, 721, 1440] -> [mslp, u10, v10, t2m]
        # 粗化后数据 shape: [4, 61, 120] (如果是Daily文件)
        
        mslp = data[0]
        u10 = data[1]
        v10 = data[2]
        t2m = data[3]
        
        # 获取纬度维度
        nlat = data.shape[1]
        
        # 根据数据分辨率调整索引
        if nlat == 721:
            # 原始0.25度分辨率
            lat_idx_n = lat_idx_north
            lat_idx_s = lat_idx_south
        elif nlat == 61:
            # 粗化到3度分辨率 (61层 = 180/3 + 1)
            lat_idx_n = int((90 - LAT_NORTH) / 3)
            lat_idx_s = int((90 - LAT_SOUTH) / 3)
        else:
            # 其他分辨率，使用比例
            scale = nlat / 721.0
            lat_idx_n = int(lat_idx_north * scale)
            lat_idx_s = int(lat_idx_south * scale)
        
        # 提取区域数据 (北纬60到10，索引从小到大)
        t2m_region = t2m[lat_idx_n:lat_idx_s, :]
        u10_region = u10[lat_idx_n:lat_idx_s, :]
        v10_region = v10[lat_idx_n:lat_idx_s, :]
        mslp_region = mslp[lat_idx_n:lat_idx_s, :]
        
        # 转换单位
        t2m_c = t2m_region - 273.15  # K -> °C
        mslp_hpa = mslp_region / 100.0  # Pa -> hPa
        wind_speed = np.sqrt(u10_region**2 + v10_region**2)  # m/s
        
        # 区域平均
        mean_t2m = np.mean(t2m_c)
        mean_wind = np.mean(wind_speed)
        mean_mslp = np.mean(mslp_hpa)
        
        return mean_t2m, mean_wind, mean_mslp
        
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def get_era5_monthly_data(start_year, sorted_months):
    """获取ERA5月平均数据用于对比"""
    try:
        import xarray as xr
        import cdsapi
    except ImportError:
        print("[WARN] xarray or cdsapi not installed, skipping ERA5 data")
        return None, None, None, None
    
    # ERA5数据通常存放在本地或需要下载
    # 尝试先检查本地是否有ERA5数据
    project_root = find_project_root()
    data_disk = os.path.join(project_root, 'autodl-tmp')
    if os.path.exists(data_disk):
        base_dir = data_disk
    else:
        base_dir = project_root
    
    era5_dir = os.path.join(base_dir, 'Input', 'ERA5_monthly')
    os.makedirs(era5_dir, exist_ok=True)
    
    print(f"[INFO] Looking for ERA5 data in: {era5_dir}")
    
    # 检查是否已有ERA5数据文件
    era5_file = os.path.join(era5_dir, f'era5_monthly_{start_year}.nc')
    
    if not os.path.exists(era5_file):
        print(f"[INFO] Downloading ERA5 data for {start_year}...")
        try:
            download_era5_data(era5_file, start_year, sorted_months)
        except Exception as e:
            print(f"[WARN] Failed to download ERA5 data: {e}")
            return None, None, None, None
    
    # 加载ERA5数据
    try:
        ds = xr.open_dataset(era5_file)
        print(f"[DEBUG] ERA5 dataset variables: {list(ds.data_vars)}")
        print(f"[DEBUG] ERA5 dataset dimensions: {list(ds.dims)}")
        print(f"[DEBUG] ERA5 dataset coords: {list(ds.coords)}")
        
        lat_coord = ds['latitude']
        lat_slice = slice(LAT_NORTH, LAT_SOUTH) if lat_coord.values[0] > lat_coord.values[-1] else slice(LAT_SOUTH, LAT_NORTH)
        print(f"[DEBUG] ERA5 latitude order: {lat_coord.values[0]} -> {lat_coord.values[-1]}, use slice {lat_slice}")
        
        # 提取变量（假设变量名为 t2m, u10, v10, mslp）
        t2m_era5 = []
        wind_era5 = []
        mslp_era5 = []
        years_era5 = []
        
        # 确定时间变量名称
        time_var = 'valid_time' if 'valid_time' in ds.dims else 'time'
        print(f"[DEBUG] Using time variable: {time_var}")
        
        for year, month in sorted_months:
            try:
                # 时间范围过滤
                if year < TIME_START or year > TIME_END:
                    continue
                
                # 从ERA5数据中提取该月的值
                month_str = f'{year}-{month:02d}'
                
                # 尝试使用valid_time或time进行选择
                try:
                    if time_var == 'valid_time':
                        month_data = ds.sel(valid_time=month_str)
                    else:
                        month_data = ds.sel(time=month_str)
                except KeyError:
                    # 如果精确匹配失败，尝试模糊匹配
                    time_coord = ds[time_var]
                    matching_times = [t for t in time_coord.values 
                                     if str(t).startswith(month_str)]
                    if matching_times:
                        if time_var == 'valid_time':
                            month_data = ds.sel(valid_time=matching_times)
                        else:
                            month_data = ds.sel(time=matching_times)
                    else:
                        print(f"[WARN] No data found for {month_str}")
                        continue
                
                # 计算该月中旬的年份值
                mid_date = datetime(year, month, 15)
                year_start = datetime(year, 1, 1)
                try:
                    year_end = datetime(year + 1, 1, 1)
                except:
                    year_end = datetime(year, 12, 31)
                days_in_year = (year_end - year_start).days
                year_value = year + (mid_date - year_start).days / days_in_year
                years_era5.append(year_value)
                
                # 提取NH(10N-60N)的平均值
                t2m_field = month_data['t2m'].sel(latitude=lat_slice)
                u10_field = month_data['u10'].sel(latitude=lat_slice)
                v10_field = month_data['v10'].sel(latitude=lat_slice)
                mslp_field = month_data['msl'].sel(latitude=lat_slice)
                
                t2m_nh = t2m_field.mean().values
                u10_nh = u10_field.mean().values
                v10_nh = v10_field.mean().values
                mslp_nh = mslp_field.mean().values
                
                t2m_era5.append(float(t2m_nh - 273.15))  # K -> °C
                wind_era5.append(float(np.sqrt(u10_nh**2 + v10_nh**2)))  # m/s
                mslp_era5.append(float(mslp_nh / 100.0))  # Pa -> hPa
            except Exception as e:
                print(f"[DEBUG] Failed to extract {year}-{month}: {e}")
                continue
        
        if not t2m_era5:
            print("[WARN] No ERA5 data extracted")
            return None, None, None, None
        
        print(f"[INFO] Successfully extracted {len(t2m_era5)} months of ERA5 data")
        return t2m_era5, wind_era5, mslp_era5, years_era5
        
    except Exception as e:
        print(f"[ERROR] Failed to load ERA5: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def download_era5_data(output_file, start_year, sorted_months):
    """下载ERA5月平均数据"""
    try:
        import cdsapi
    except ImportError:
        print("[WARN] cdsapi not installed. Install with: pip install cdsapi")
        raise
    
    client = cdsapi.Client()
    
    # 准备请求参数
    end_year = sorted_months[-1][0]
    months = list(set([m[1] for m in sorted_months]))
    
    request = {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': ['2m_temperature', '10m_u_component_of_wind', 
                     '10m_v_component_of_wind', 'mean_sea_level_pressure'],
        'year': [str(y) for y in range(start_year, end_year + 1)],
        'month': [f'{m:02d}' for m in months],
        'time': '00:00',
        'format': 'netcdf'
    }
    
    print(f"[INFO] Downloading ERA5 from {start_year} to {end_year}...")
    client.retrieve('reanalysis-era5-single-levels-monthly-means', 
                    request, output_file)
    print(f"[INFO] ERA5 data saved to: {output_file}")

def draw_nh_time_series(start_year=None):
    """绘制北半球时间序列"""
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")
    
    # 自动推断start_year
    if start_year is None:
        start_year = find_start_year_from_dirs(project_root)
        if start_year is None:
            print("[ERROR] Cannot determine start year. Please specify manually.")
            return
    
    print(f"[INFO] Using start year: {start_year}")
    
    # 查找所有文件
    files_dict = find_all_daily_files(project_root, start_year)
    
    if not files_dict:
        print("[ERROR] No daily surface files found!")
        return
    
    # 按日期排序
    sorted_dates = sorted(files_dict.keys())
    print(f"[INFO] Found {len(sorted_dates)} daily files")
    
    # 按月聚合，计算月平均值
    from collections import defaultdict
    monthly_data = defaultdict(lambda: {'t2m': [], 'wind': [], 'mslp': []})
    
    print(f"\n[INFO] Processing {len(sorted_dates)} daily files...")
    for i, file_date in enumerate(sorted_dates):
        file_path = files_dict[file_date]
        mean_t2m, mean_wind, mean_mslp = load_and_compute_regional_mean(
            file_path, LAT_IDX_NORTH, LAT_IDX_SOUTH
        )
        
        if mean_t2m is not None:
            # 按月份分组
            month_key = (file_date.year, file_date.month)
            monthly_data[month_key]['t2m'].append(mean_t2m)
            monthly_data[month_key]['wind'].append(mean_wind)
            monthly_data[month_key]['mslp'].append(mean_mslp)
            
            if (i + 1) % 100 == 0:
                print(f"[INFO] Processed {i + 1}/{len(sorted_dates)} files")
        else:
            print(f"[WARN] Skipping {file_date} due to processing error")
    
    if not monthly_data:
        print("[ERROR] No valid data to plot!")
        return
    
    # 计算月平均
    sorted_months = sorted(monthly_data.keys())
    t2m_series = []
    wind_series = []
    mslp_series = []
    years = []
    
    print(f"\n[INFO] Computing monthly averages for {len(sorted_months)} months...")
    for year, month in sorted_months:
        # 时间范围过滤
        if year < TIME_START or year > TIME_END:
            continue
        
        # 计算该月的平均值
        t2m_avg = np.mean(monthly_data[(year, month)]['t2m'])
        wind_avg = np.mean(monthly_data[(year, month)]['wind'])
        mslp_avg = np.mean(monthly_data[(year, month)]['mslp'])
        
        t2m_series.append(t2m_avg)
        wind_series.append(wind_avg)
        mslp_series.append(mslp_avg)
        
        # 计算该月中旬的年份值 (用于x轴坐标)
        mid_date = datetime(year, month, 15)
        year_start = datetime(year, 1, 1)
        try:
            year_end = datetime(year + 1, 1, 1)
        except:
            year_end = datetime(year, 12, 31)
        days_in_year = (year_end - year_start).days
        year_value = year + (mid_date - year_start).days / days_in_year
        years.append(year_value)
    
    print(f"[INFO] Data range: T2m [{min(t2m_series):.2f}, {max(t2m_series):.2f}]°C")
    print(f"[INFO]             Wind [{min(wind_series):.2f}, {max(wind_series):.2f}] m/s")
    print(f"[INFO]             MSLP [{min(mslp_series):.2f}, {max(mslp_series):.2f}] hPa")
    
    # 获取ERA5对比数据
    print(f"\n[INFO] Loading ERA5 data for comparison...")
    era5_t2m, era5_wind, era5_mslp, era5_years = get_era5_monthly_data(
        start_year, sorted_months
    )
    
    if era5_t2m is None:
        print("[WARN] ERA5 data not available, showing model data only")
        era5_t2m = era5_wind = era5_mslp = None
    else:
        # 调试输出：检查数据长度
        print(f"[DEBUG] Pangu data points: {len(years)}")
        print(f"[DEBUG] ERA5 data points: {len(era5_years)}")
        print(f"[DEBUG] Pangu year range: {years[0]:.2f} - {years[-1]:.2f}")
        print(f"[DEBUG] ERA5 year range: {era5_years[0]:.2f} - {era5_years[-1]:.2f}")
        
        # 如果长度不匹配，进行裁剪或对齐
        if len(years) != len(era5_years):
            print(f"[WARN] Data length mismatch! Pangu={len(years)}, ERA5={len(era5_years)}")
            # 仅保留共同的时间范围
            min_len = min(len(years), len(era5_years))
            years = years[:min_len]
            t2m_series = t2m_series[:min_len]
            wind_series = wind_series[:min_len]
            mslp_series = mslp_series[:min_len]
            era5_years = era5_years[:min_len]
            t2m_era5 = t2m_era5[:min_len]
            wind_era5 = wind_era5[:min_len]
            mslp_era5 = mslp_era5[:min_len]
            print(f"[INFO] Trimmed to {min_len} common data points")
    
    # === 绘图：3个子图竖排 ===
    fig, axes = plt.subplots(3, 1, figsize=FIG_SIZE)
    
    # 子图1：2m Temperature
    axes[0].plot(years, t2m_series, color=COLOR_T2M, linewidth=2, marker='o', 
                 markersize=2, label='Pangu')
    axes[0].set_ylabel('T2m (°C)', color=COLOR_T2M, fontsize=11)
    axes[0].tick_params(axis='y', labelcolor=COLOR_T2M)
    axes[0].grid(True, linestyle='--', alpha=0.3)
    
    if era5_t2m is not None:
        ax0_right = axes[0].twinx()
        ax0_right.plot(era5_years, era5_t2m, color='#990000', linewidth=2, marker='o', 
                       markersize=2)
        ax0_right.set_ylabel('ERA5 T2m (°C)', color='#990000', fontsize=11)
        ax0_right.tick_params(axis='y', labelcolor='#990000')
        
        # 统一坐标轴范围，以ERA5为准
        era5_min, era5_max = min(era5_t2m), max(era5_t2m)
        era5_range = era5_max - era5_min
        padding = era5_range * 0.1
        ax0_right.set_ylim(era5_min - padding, era5_max + padding)
        axes[0].set_ylim(era5_min - padding, era5_max + padding)
    
    # 子图2：10m Wind Speed
    axes[1].plot(years, wind_series, color=COLOR_WIND, linewidth=2, marker='s', 
                 markersize=2, label='Pangu')
    axes[1].set_ylabel('Wind (m/s)', color=COLOR_WIND, fontsize=11)
    axes[1].tick_params(axis='y', labelcolor=COLOR_WIND)
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    if era5_wind is not None:
        ax1_right = axes[1].twinx()
        ax1_right.plot(era5_years, era5_wind, color='#003399', linewidth=2, marker='s', 
                       markersize=2)
        ax1_right.set_ylabel('ERA5 Wind (m/s)', color='#003399', fontsize=11)
        ax1_right.tick_params(axis='y', labelcolor='#003399')
        
        # 统一坐标轴范围，以ERA5为准
        era5_min, era5_max = min(era5_wind), max(era5_wind)
        era5_range = era5_max - era5_min
        padding = era5_range * 0.1
        ax1_right.set_ylim(era5_min - padding, era5_max + padding)
        axes[1].set_ylim(era5_min - padding, era5_max + padding)
    
    # 子图3：MSLP
    axes[2].plot(years, mslp_series, color=COLOR_MSLP, linewidth=2, marker='^', 
                 markersize=2, label='Pangu')
    axes[2].set_ylabel('MSLP (hPa)', color=COLOR_MSLP, fontsize=11)
    axes[2].set_xlabel('Year', fontsize=11)
    axes[2].tick_params(axis='y', labelcolor=COLOR_MSLP)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    if era5_mslp is not None:
        ax2_right = axes[2].twinx()
        ax2_right.plot(era5_years, era5_mslp, color='#1a4d00', linewidth=2, marker='^', 
                       markersize=2)
        ax2_right.set_ylabel('ERA5 MSLP (hPa)', color='#1a4d00', fontsize=11)
        ax2_right.tick_params(axis='y', labelcolor='#1a4d00')
        
        # 统一坐标轴范围，以ERA5为准
        era5_min, era5_max = min(era5_mslp), max(era5_mslp)
        era5_range = era5_max - era5_min
        padding = era5_range * 0.1
        ax2_right.set_ylim(era5_min - padding, era5_max + padding)
        axes[2].set_ylim(era5_min - padding, era5_max + padding)
    
    # 主标题
    fig.suptitle(f'Pangu NH({LAT_SOUTH}N-{LAT_NORTH}N) Monthly Mean Time Series',
                 x=0.08, y=0.92, ha='left', fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # 保存 - 遵循longterm_integration.py的逻辑
    data_disk = os.path.join(project_root, 'autodl-tmp')
    if os.path.exists(data_disk):
        base_output = data_disk
    else:
        base_output = project_root
    
    output_dir = os.path.join(base_output, 'Run-output-png', 'Pangu', f'Longterm_{start_year}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = f'NH_Time_Series_T2m_Wind_MSLP_{start_year}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"\n[INFO] Saving time series figure to: {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Done.")

if __name__ == "__main__":
    import sys
    
    # 使用配置的参数
    project_root = PROJECT_ROOT
    start_year = START_YEAR
    
    # 如果没有配置，才尝试从命令行参数读取
    if len(sys.argv) > 1:
        if project_root is None and ('/' in sys.argv[1] or '\\' in sys.argv[1]):
            project_root = sys.argv[1]
        if start_year is None:
            try:
                start_year = int(sys.argv[1] if len(sys.argv) == 2 else sys.argv[2])
            except:
                pass
    
    if project_root is None:
        print("[INFO] PROJECT_ROOT not specified, auto-detecting...")
        project_root_found = find_project_root()
        project_root = project_root_found
    else:
        print(f"[INFO] Using configured PROJECT_ROOT: {project_root}")
    
    print(f"[INFO] Project root: {project_root}")
    
    if start_year is None:
        start_year = find_start_year_from_dirs(project_root)
        if start_year is None:
            print("[ERROR] Cannot determine START_YEAR. Please specify in the script.")
            print("[INFO] Edit the script and set START_YEAR = 1950 (or other year)")
            sys.exit(1)
    
    # 直接调用绘图函数
    draw_nh_time_series(start_year)
