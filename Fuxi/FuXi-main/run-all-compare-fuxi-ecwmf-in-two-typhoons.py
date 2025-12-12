#!/usr/bin/env python
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
from cdsapi import Client
from ecmwfapi import ECMWFDataServer
import cfgrib
import subprocess

# ---------------------- 配置参数 ----------------------
TY_PATH = os.path.join("two-typhoon-csv")  # typhoon csv 目录
MODEL_DIR = os.path.join("model")
# 输出路径：放在 two-typhoon-csv/new_ 下，文件名为 new_原文件名
NEW_DIR = os.path.join("two-typhoon-csv", "new_")
ERA5_CACHE_DIR = "raw_input"  # ERA5数据缓存目录
# ------------------------------------------------------

def read_typhoon_data(file_path):
    return pd.read_csv(file_path)

def get_start_time(df):
    start_time = pd.to_datetime(df['ISO_TIME'].iloc[0])
    # 确保基准时次为 0 时或 12 时
    if start_time.hour not in [0, 12]:
        if start_time.hour < 6:
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start_time = start_time.replace(hour=12, minute=0, second=0, microsecond=0)
        print(f"调整台风起始基准时次为: {start_time}")
    return start_time

def time_encoding(init_time, total_step, freq=6):
    # 根据预报起始时刻生成时间编码
    tembs = []
    for i in range(total_step):
        offsets = [pd.Timedelta(hours=t * freq) for t in [i-2, i-1, i]]
        periods = [pd.Period((init_time + off), 'h') for off in offsets]
        ratios = [(p.day_of_year/366, p.hour/24) for p in periods]
        arr = np.array(ratios, dtype=np.float32)
        temb = np.concatenate([np.sin(arr), np.cos(arr)]).reshape(1, -1)
        tembs.append(temb)
    return np.stack(tembs)

def run_fuxi_model(csv_file):
    """
    调用 download_and_make_era5.py 生成初始场数据，并通过命令行调用 fuxi.py 进行模型推理，
    预报基准时次取自 csv_file 中的起始时间及其六小时前。
    同时根据 CSV 文件最后一个时刻计算预报步长（以6小时为步长）避免过度运行。
    """
    df_typhoon = read_typhoon_data(csv_file)
    start_time = get_start_time(df_typhoon)
    # 计算 CSV 中最后一个时刻与起始时刻的差，为预报有效时长
    end_time = pd.to_datetime(df_typhoon['ISO_TIME'].iloc[-1])
    total_hours = (end_time - start_time).total_seconds() / 3600.0
    # 每6小时一个步长
    num_steps = int(total_hours // 6)
    if num_steps <= 0:
        num_steps = 1  # 至少运行1步预报

    input_file = os.path.join("input", "input.nc")
    times_arg = f"{start_time.strftime('%Y%m%d-%H')},{(start_time - timedelta(hours=6)).strftime('%Y%m%d-%H')}"
    
    cmd = ["python", r"FuXi-main/download_and_make_era5.py",
           "--times", times_arg,
           "--data_dir", "raw_input",
           "--output", input_file]
    print("调用 download_and_make_era5.py 生成初始场数据...")
    subprocess.check_call(cmd)
    
    if num_steps <= 20:
        steps1, steps2, steps3 = num_steps, 0, 0
    elif num_steps <= 40:
        steps1, steps2, steps3 = 20, num_steps-20, 0
    else:
        steps1, steps2, steps3 = 20, 20, num_steps-40

    cmd_fuxi = [
        "python", r"FuXi-main/fuxi.py",
        "--model", MODEL_DIR,
        "--input", input_file,
        "--num_steps", str(steps1), str(steps2), str(steps3),
        "--save_dir", "output"
    ]
    print(f"通过命令行调用 fuxi.py 进行模型推理，预报步长为 {num_steps}（共 {num_steps*6} 小时）...")
    subprocess.check_call(cmd_fuxi)

def download_ecmwf_forecast(start_time):
    """
    以 CSV 中的基准时刻作为 ECMWF 预报的基准（仅一个日期和时间），
    并请求每6小时的预报（从0h开始），确保预报有效时间与 CSV 中的时刻对应。
    """
    req_params = {
        "class": "ti",
        "dataset": "tigge",
        # 使用单一的基准日期
        "date": start_time.strftime('%Y-%m-%d'),
        # 使用 CSV 中的小时作为基准时间，例如 "00:00:00"
        "time": start_time.strftime('%H:00:00'),
        "expver": "prod",
        "grid": "0.5/0.5",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "151",
        # 请求每6小时的预报步长，从0h开始
        "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168",
        "type": "fc",
        "target": os.path.join("raw_input", "ecmwf_forecast.grib")
    }
    os.makedirs("output", exist_ok=True)
    os.makedirs("raw_input", exist_ok=True)
    
    print("开始下载 ECMWF 预报数据...")
    ECMWFDataServer().retrieve(req_params)
    print("ECMWF预报数据下载完成。")

def get_min_pressure_from_numpy(data_array, lat_arr, lon_arr, center_lat, center_lon):
    """
    在 (center_lat, center_lon)±4° 范围内提取最低值及对应经纬度，
    作为遗留接口使用。
    """
    lat_mask = (lat_arr >= (center_lat - 4)) & (lat_arr <= (center_lat + 4))
    lon_mask = (lon_arr >= (center_lon - 4)) & (lon_arr <= (center_lon + 4))
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    if len(lat_indices) == 0 or len(lon_indices) == 0:
        print(f"经纬度范围 ({center_lat}±4°, {center_lon}±4°) 内无数据")
        return {'MinPressure': np.nan, 'Lat': center_lat, 'Lon': center_lon}
    
    subset = data_array[np.ix_(lat_indices, lon_indices)]
    if subset.size == 0:
        print("选定区域提取为空")
        return {'MinPressure': np.nan, 'Lat': center_lat, 'Lon': center_lon}
        
    min_val = float(subset.min())
    sub_idx = np.unravel_index(np.argmin(subset), subset.shape)
    sel_lat = float(lat_arr[lat_indices[sub_idx[0]]])
    sel_lon = float(lon_arr[lon_indices[sub_idx[1]]])
    return {'MinPressure': min_val, 'Lat': sel_lat, 'Lon': sel_lon}

# 新增辅助函数：在指定中心±search_range范围内寻找最低气压中心
def get_min_pressure_within_range(data_array, lat_arr, lon_arr, center_lat, center_lon, search_range):
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

# 新增辅助函数：两步链式搜索台风中心，在±5°范围内先搜索一次，再以第一次结果继续搜索
def find_typhoon_center(data_array, lat_arr, lon_arr, init_lat, init_lon):
    first_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, init_lat, init_lon, 3)
    second_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, first_search['Lat'], first_search['Lon'], 3)
    return second_search

def extract_fuxi_forecast(iso_time, best_lat, best_lon, prev_center=None):
    """
    从 FuXi 模型预报数据中提取最低气压点。
    若 prev_center 不为 None，则以其作为链式搜索的初始中心，否则使用 CSV 中提供的 (best_lat, best_lon)。
    """
    df_typhoon = read_typhoon_data(current_csv)
    start_time_csv = get_start_time(df_typhoon)
    
    forecast_hour = int((iso_time - start_time_csv).total_seconds() // 3600)
    file_name = f"{forecast_hour+6:03d}.nc"
    file_path = os.path.join("output", file_name)
    
    if not os.path.exists(file_path):
        print(f"未找到 FuXi 模型预报文件: {file_name}")
        return {'MinPressure': np.nan, 'Lat': prev_center[0], 'Lon': prev_center[1]}
    
    ds = xr.open_dataset(file_path)
    pressure_var = "__xarray_dataarray_variable__"
    # 假设文件中存在 pressure_var 和 'MSL' 层次
    msl_index = np.where(ds['level'].values == 'MSL')[0][0]
    pressure_data = ds[pressure_var][0, 0, msl_index, :, :].values / 100
    print("FuXi 模型预报时刻", iso_time, "pressure range:", pressure_data.min(), pressure_data.max())
    lat_arr = ds['lat'].values
    lon_arr = ds['lon'].values
    ds.close()
    
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    # 使用两步链式搜索确定台风中心
    result = find_typhoon_center(pressure_data, lat_arr, lon_arr, init_lat, init_lon)
    print(f"FuXi {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=None):
    """
    从 ECMWF GRIB 文件中提取最低气压点，链式搜索机制与 FuXi 一致。
    """
    grib_path = os.path.join("raw_input", "ecmwf_forecast.grib")
    ds = xr.open_dataset(grib_path, engine='cfgrib', decode_timedelta=False)
    
    valid_times = pd.to_datetime(ds["valid_time"].values, errors="coerce")
    idx = int(np.argmin(np.abs(valid_times - iso_time)))
    
    msl = ds["msl"].isel(step=idx)
    # 尝试获取纬度和经度坐标（优先使用 "latitude" 和 "longitude"）
    lats = msl["latitude"].values if "latitude" in msl.coords else msl["lat"].values
    lons = msl["longitude"].values if "longitude" in msl.coords else msl["lon"].values
    
    pressure_data = msl.values / 100  # 转换单位
    ds.close()
    
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    result = find_typhoon_center(pressure_data, lats, lons, init_lat, init_lon)
    print(f"ECMWF {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def merge_data_to_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # 按 ISO_TIME 升序排序，确保时间顺序
    df.sort_values("ISO_TIME", inplace=True)
    
    # 用于存储各行的结果，保持原索引关联
    fuxi_results = {}
    ecmwf_results = {}
    
    # 按台风名称分组（或根据 SID 分组），确保不同台风各自链式搜索不混淆
    for name, group in df.groupby("NAME"):
        # 每个台风单独初始化链式中心
        prev_center_fuxi = None
        prev_center_ecmwf = None
        # 注意：group 里的行索引仍为原 DataFrame 的索引
        for idx, row in group.iterrows():
            iso_time = pd.to_datetime(row['ISO_TIME'])
            best_lat, best_lon = row['LAT'], row['LON']
            res_fuxi = extract_fuxi_forecast(iso_time, best_lat, best_lon, prev_center=prev_center_fuxi)
            # 保存时记录原索引，便于回填
            fuxi_results[idx] = res_fuxi
            if not np.isnan(res_fuxi['Lat']) and not np.isnan(res_fuxi['Lon']):
                prev_center_fuxi = (res_fuxi['Lat'], res_fuxi['Lon'])
            
            res_ecmwf = extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=prev_center_ecmwf)
            ecmwf_results[idx] = res_ecmwf
            if not np.isnan(res_ecmwf['Lat']) and not np.isnan(res_ecmwf['Lon']):
                prev_center_ecmwf = (res_ecmwf['Lat'], res_ecmwf['Lon'])
    
    # 将结果按照 DataFrame 原索引顺序排序后回填到 DataFrame 中
    df['FuxiMinPressure'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('MinPressure', np.nan))
    df['FuxiMinLat'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('Lat', np.nan))
    df['FuxiMinLon'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('Lon', np.nan))
    df['ECMWFMinPressure'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('MinPressure', np.nan))
    df['ECMWFMinLat'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lat', np.nan))
    df['ECMWFMinLon'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lon', np.nan))
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"合并并解读后的数据已保存到: {output_csv}")

def process_workflow(csv_file):
    """
    针对单个 csv_file 执行流程：
      - 运行 FuXi 模型
      - 下载 ECMWF 预报数据
      - 合并解读结果并输出新 CSV（输出文件名为 new_原文件名）
    """
    global current_csv
    current_csv = csv_file
    df_typhoon = read_typhoon_data(csv_file)
    start_time = get_start_time(df_typhoon)
    print(f"处理文件 {csv_file}，台风起始时间: {start_time}")
    
    run_fuxi_model(csv_file)
    download_ecmwf_forecast(start_time)
    
    base_name = os.path.basename(csv_file)
    output_csv = os.path.join(NEW_DIR, "new_" + base_name)
    merge_data_to_csv(csv_file, output_csv)

def process_all_csv():
    """
    遍历 two-typhoon-csv 目录下所有 csv 文件（排除 new_ 子目录），分别执行流程
    """
    for file in os.listdir(TY_PATH):
        if file.endswith(".csv") and not file.startswith("new_"):
            csv_file = os.path.join(TY_PATH, file)
            process_workflow(csv_file)

def main():
    process_all_csv()

if __name__ == "__main__":
    main()