import os
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import cfgrib
from ecmwfapi import ECMWFDataServer
from datetime import datetime, timedelta

from get_era5_data import get_era5_data
from get_gfs_data import get_gfs_data
from draw_mslp_and_wind import draw_mslp_and_wind
from draw_path import draw_path

# 全局路径配置（将在 main 中初始化）
data_source = None
output_dir = None
output_image_path = None
output_path = None
input_path = None
# 使用的经纬度范围（与 Pangu 模型一致）
lat_min = None
lat_max = None
lon_min = None
lon_max = None
# 当前处理的 typhoon CSV 文件路径（用于提取基准时间）
current_csv = None

def run_pangu(model_type, data_source, run_times, base_datetime, output_image_path='output_png'):
    """
    运行 Pangu 模型，生成预测结果并保存为图片，
    返回一个字典，映射预报时次（小时）到对应的 output_surface 文件名。
    """
    print("Loading ONNX model...")
    model = onnx.load(f'weight/pangu_weather_{model_type}.onnx')
    
    print("Setting ONNX Runtime options...")
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    
    print("Setting CUDA provider options...")
    cuda_provider_options = {
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }
    
    print("Initializing ONNX Runtime session...")
    ort_session = ort.InferenceSession(
        f'weight/pangu_weather_{model_type}.onnx',
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    
    current_datetime = base_datetime  # 格式：YYYYMMDDHH 字符串
    files_dict = {}
    
    for i in range(run_times + 1):
        hour = model_type * i

        output_upper_filename = os.path.join(output_path, f'output_upper_{current_datetime}+{hour:02d}h_{data_source}.npy')
        output_surface_filename = os.path.join(output_path, f'output_surface_{current_datetime}+{hour:02d}h_{data_source}.npy')
        next_hour = hour + model_type
        next_output_upper_filename = os.path.join(output_path, f'output_upper_{current_datetime}+{next_hour:02d}h_{data_source}.npy')
        next_output_surface_filename = os.path.join(output_path, f'output_surface_{current_datetime}+{next_hour:02d}h_{data_source}.npy')

        output_png_name = f'{output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{hour:02d}h.png'
        next_output_png_name = f'{output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{next_hour:02d}h.png'
        output_path_png_name = f'{output_image_path}/path_{data_source}_{current_datetime}+{next_hour:02d}h.png'

        if i == 0:
            print("Loading initial input data...")
            input_upper = np.load(os.path.join(input_path, 'input_upper.npy')).astype(np.float32)
            input_surface = np.load(os.path.join(input_path, 'input_surface.npy')).astype(np.float32)
            
            print("Drawing initial MSLP and wind field...")
            draw_mslp_and_wind(current_datetime, model_type, run_times,
                               os.path.join(input_path, 'input_surface.npy'),
                               output_png_name, hour,
                               lon_min, lon_max, lat_min, lat_max, data_source)
            print(f"Saved initial image to {output_png_name}")

            if os.path.exists(next_output_upper_filename) and os.path.exists(next_output_surface_filename):
                print(f"Output files {next_output_upper_filename} and {next_output_surface_filename} already exist. Skipping inference.")
            else:
                print(f"Running inference from {hour:02d}h to {next_hour:02d}h...")
                output, output_surface = ort_session.run(None, {'input': input_upper, 'input_surface': input_surface})
                print(f"Saving results to {next_output_upper_filename} and {next_output_surface_filename}...")
                np.save(next_output_upper_filename, output)
                np.save(next_output_surface_filename, output_surface)
                print("Drawing MSLP and wind field...")
                draw_mslp_and_wind(current_datetime, model_type, run_times,
                                   next_output_surface_filename, next_output_png_name, next_hour,
                                   lon_min, lon_max, lat_min, lat_max, data_source)
                print(f"Saved image to {next_output_png_name}")

        else:
            
            if os.path.exists(next_output_upper_filename) and os.path.exists(next_output_surface_filename):
                print(f"Output files {next_output_upper_filename} and {next_output_surface_filename} already exist. Skipping inference.")
            else:
                input_upper = np.load(output_upper_filename).astype(np.float32)
                input_surface = np.load(output_surface_filename).astype(np.float32)
                print(f"Running inference from {hour:02d}h to {next_hour:02d}h...")
                output, output_surface = ort_session.run(None, {'input': input_upper, 'input_surface': input_surface})
                print(f"Saving results to {next_output_upper_filename} and {next_output_surface_filename}...")
                np.save(next_output_upper_filename, output)
                np.save(next_output_surface_filename, output_surface)
            print("Drawing MSLP and wind field...")
            draw_mslp_and_wind(current_datetime, model_type, run_times,
                               next_output_surface_filename, next_output_png_name, next_hour,
                               lon_min, lon_max, lat_min, lat_max, data_source)
            print(f"Saved image to {next_output_png_name}")
        
        files_dict[next_hour] = next_output_surface_filename
        
    print('Pangu 模型运行完毕！')
    print("生成文件字典:", files_dict)
    return files_dict

def download_ecmwf_forecast(start_time):
    """
    使用 CSV 中的基准时刻下载 ECMWF 预报数据（每6小时的时次）
    """
    req_params = {
        "class": "ti",
        "dataset": "tigge",
        "date": start_time.strftime('%Y-%m-%d'),
        "time": start_time.strftime('%H:00:00'),
        "expver": "prod",
        "grid": "0.5/0.5",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "151",
        "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168",
        "type": "fc",
        "target": os.path.join("raw_input", "ecmwf_forecast.grib")
    }
    os.makedirs("raw_input", exist_ok=True)
    
    print("开始下载 ECMWF 预报数据...")
    ECMWFDataServer().retrieve(req_params)
    print("ECMWF预报数据下载完成。")

def get_start_time(df):
    """
    从 typhoon CSV 中提取起始时刻，确保时次为 0 时或 12 时
    """
    start_time = pd.to_datetime(df['ISO_TIME'].iloc[0])
    if start_time.hour not in [0, 12]:
        if start_time.hour < 6:
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_time = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
        print(f"调整台风起始基准时次为: {start_time}")
    return start_time

# ------------------ 提取气压中心算法 ------------------
def get_min_pressure_within_range(data_array, lat_arr, lon_arr, center_lat, center_lon, search_range):
    """
    在 (center_lat, center_lon)±search_range 范围内搜索最低气压及对应经纬度
    """
    lat_mask = (lat_arr >= (center_lat - search_range)) & (lat_arr <= (center_lat + search_range))
    lon_mask = (lon_arr >= (center_lon - search_range)) & (lon_arr <= (center_lon + search_range))
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        print(f"经纬度范围 ({center_lat}±{search_range}°, {center_lon}±{search_range}°) 内无数据")
        return {'MinPressure': np.nan, 'Lat': center_lat, 'Lon': center_lon}
    subset = data_array[np.ix_(lat_indices, lon_indices)]
    if subset.size == 0:
        print("选定区域提取为空")
        return {'MinPressure': np.nan, 'Lat': center_lat, 'Lon': center_lon}
    min_val = float(subset.min())
    sub_idx = np.unravel_index(np.argmin(subset), subset.shape)
    found_lat = float(lat_arr[lat_indices[sub_idx[0]]])
    found_lon = float(lon_arr[lon_indices[sub_idx[1]]])
    return {'MinPressure': min_val, 'Lat': found_lat, 'Lon': found_lon}

def find_typhoon_center(data_array, lat_arr, lon_arr, init_lat, init_lon):
    """
    两步链式搜索台风中心
    """
    first_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, init_lat, init_lon, 2)
    second_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, first_search['Lat'], first_search['Lon'], 2)
    return second_search

def extract_pangu_forecast(iso_time, best_lat, best_lon, base_time, data_source, prev_center=None, files_dict=None):
    """
    从 Pangu 模型预报输出（npy 文件）中提取最低气压中心，
    使用两步链式搜索方法（与 ECMWF 保持一致）：
      - 使用 np.linspace 构造全局经纬度网格
      - 截取指定区域，使用 find_typhoon_center 进行链式搜索
    """
    forecast_hour = int((iso_time - base_time).total_seconds() // 3600)
    if files_dict is not None and forecast_hour in files_dict:
        file_name = files_dict[forecast_hour]
    else:
        base_time_str = base_time.strftime("%Y%m%d%H")
        file_name = os.path.join(output_path, f'output_surface_{base_time_str}+{forecast_hour:02d}h_{data_source}.npy')
    print(file_name)
    if not os.path.exists(file_name):
        print(f"未找到 Pangu 模型预报文件: {file_name}")
        return {'MinPressure': np.nan, 'Lat': best_lat, 'Lon': best_lon}
    
    # 加载压力数据（注意：气压场数据存储在 npy 文件的第一层），转换为 hPa
    data = np.load(file_name)[0] / 100
    ny, nx = data.shape
    # 构造全局网格，与 draw_mslp_and_wind 保持一致
    full_lons = np.linspace(0, 360, nx)
    full_lats = np.linspace(90, -90, ny)
    
    # 截取指定感兴趣区域
    lon_mask = (full_lons >= lon_min) & (full_lons <= lon_max)
    lat_mask = (full_lats <= lat_max) & (full_lats >= lat_min)
    if not lon_mask.any() or not lat_mask.any():
        print(f"指定区域({lon_min}~{lon_max}, {lat_min}~{lat_max})超出数据范围")
        return {'MinPressure': np.nan, 'Lat': best_lat, 'Lon': best_lon}
    region_data = data[np.ix_(np.where(lat_mask)[0], np.where(lon_mask)[0])]
    region_lons = full_lons[lon_mask]
    region_lats = full_lats[lat_mask]
    
    # 使用两步链式搜索确定台风中心
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    result = find_typhoon_center(region_data, region_lats, region_lons, init_lat, init_lon)
    
    print(f"Pangu {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=None):
    """
    从 ECMWF GRIB 文件中提取最低气压中心，链式搜索与 Pangu 相同
    """
    grib_path = os.path.join("raw_input", "ecmwf_forecast.grib")
    ds = xr.open_dataset(grib_path, engine='cfgrib', decode_timedelta=False)
    valid_times = pd.to_datetime(ds["valid_time"].values, errors="coerce")
    idx = int(np.argmin(np.abs(valid_times - iso_time)))
    
    msl = ds["msl"].isel(step=idx)
    if "latitude" in msl.coords:
        lats = msl["latitude"].values
    else:
        lats = msl["lat"].values
    if "longitude" in msl.coords:
        lons = msl["longitude"].values
    else:
        lons = msl["lon"].values
    pressure_data = msl.values / 100  # 单位转换为 hPa
    ds.close()
    
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    result = find_typhoon_center(pressure_data, lats, lons, init_lat, init_lon)
    print(f"ECMWF {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def merge_data_to_csv(input_csv, output_csv, files_dict=None):
    """
    读取原始台风 CSV，提取每个时次的 Pangu 与 ECMWF 最低气压中心，
    并将结果合并保存到新的 CSV 文件中。
    files_dict 用于为 Pangu 数据读取提供统一的文件列表。
    """
    df = pd.read_csv(input_csv)
    # 修改排序方式，先按台风名称，再按时间排序
    df.sort_values(["NAME", "ISO_TIME"], inplace=True)
    base_time = get_start_time(df)
    
    pangu_results = {}
    ecmwf_results = {}
    
    for name, group in df.groupby("NAME"):
        prev_center_pangu = None
        prev_center_ecmwf = None
        for idx, row in group.iterrows():
            iso_time = pd.to_datetime(row['ISO_TIME'])
            best_lat, best_lon = row['LAT'], row['LON']
            res_pangu = extract_pangu_forecast(iso_time, best_lat, best_lon, base_time, data_source,
                                               prev_center=prev_center_pangu, files_dict=files_dict)
            pangu_results[idx] = res_pangu
            if not np.isnan(res_pangu['Lat']) and not np.isnan(res_pangu['Lon']):
                prev_center_pangu = (res_pangu['Lat'], res_pangu['Lon'])
            
            res_ecmwf = extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=prev_center_ecmwf)
            ecmwf_results[idx] = res_ecmwf
            if not np.isnan(res_ecmwf['Lat']) and not np.isnan(res_ecmwf['Lon']):
                prev_center_ecmwf = (res_ecmwf['Lat'], res_ecmwf['Lon'])
    
    df['PanguMinPressure'] = df.index.map(lambda i: pangu_results.get(i, {}).get('MinPressure', np.nan))
    df['PanguMinLat'] = df.index.map(lambda i: pangu_results.get(i, {}).get('Lat', np.nan))
    df['PanguMinLon'] = df.index.map(lambda i: pangu_results.get(i, {}).get('Lon', np.nan))
    df['ECMWFMinPressure'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('MinPressure', np.nan))
    df['ECMWFMinLat'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lat', np.nan))
    df['ECMWFMinLon'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lon', np.nan))
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"合并后的数据已保存到: {output_csv}")

def process_workflow(csv_file):
    """针对单个CSV执行处理流程"""
    global current_csv
    current_csv = csv_file
    df_typhoon = pd.read_csv(csv_file)
    base_time = get_start_time(df_typhoon)
    print(f"处理地形影响台风文件 {csv_file}，基准时刻: {base_time}")
    
    # 数据下载部分保持不变
    if data_source.upper() == "ERA5":
        get_era5_data(output_dir, base_time.strftime("%Y%m%d%H"))
    elif data_source.upper() == "GFS":
        get_gfs_data(output_dir, base_time.strftime("%Y%m%d%H"))
    else:
        print("未知数据源，默认使用 ERA5。")
        get_era5_data(output_dir, base_time.strftime("%Y%m%d%H"))
    
    # 计算预报步长
    end_time = pd.to_datetime(df_typhoon['ISO_TIME'].iloc[-1])
    total_hours = (end_time - base_time).total_seconds() / 3600.0
    num_steps = int(total_hours // 6) or 1  # 保证至少1个时次
    
    # 运行Pangu模型
    files_dict = run_pangu(6, data_source, run_times=num_steps, 
                         base_datetime=base_time.strftime("%Y%m%d%H"),
                         output_image_path=output_image_path)
    
    # 下载ECMWF数据
    print("下载地形影响案例的ECMWF预报数据...")
    download_ecmwf_forecast(base_time)
    
    # 生成输出路径
    base_name = os.path.basename(csv_file)
    output_csv = os.path.join(NEW_DIR, "topo_compared_" + base_name)
    merge_data_to_csv(csv_file, output_csv, files_dict=files_dict)

def process_all_csv():
    """遍历topo-influ-csv目录处理所有CSV"""
    TY_PATH = os.path.join("topo-influ-csv")  # 修改为地形影响数据目录
    for file in os.listdir(TY_PATH):
        if file.endswith(".csv") and not file.startswith("topo_compared_"):
            csv_file = os.path.join(TY_PATH, file)
            process_workflow(csv_file)

def main():
    """主函数调整路径配置"""
    global data_source, output_dir, output_image_path, output_path, input_path
    data_source = 'ERA5'  # 可根据需要改为GFS
    output_dir = 'terrain_raw_data'  # 修改原始数据存储目录
    output_image_path = "terrain_output_png"
    output_path = os.path.join("output_data", "pangu_terrain")
    input_path = "terrain_input"
    global NEW_DIR
    NEW_DIR = os.path.join("topo-influ-csv", "compared_results")  # 修改输出目录
    
    # 初始化地理范围（根据地形影响区域调整）
    lon_min = 115   # 缩小经度范围
    lon_max = 150
    lat_min = 15    # 调整纬度范围
    lat_max = 30
    globals().update(locals())  # 设置全局范围参数

    # 创建必要目录
    os.makedirs(NEW_DIR, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    
    process_all_csv()

if __name__ == "__main__":
    main()