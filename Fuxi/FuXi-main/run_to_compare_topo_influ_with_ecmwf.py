import os
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
import cfgrib
from ecmwfapi import ECMWFDataServer
from datetime import datetime, timedelta

# 去掉了绘图相关模块的导入
# from draw_mslp_and_wind import draw_mslp_and_wind
# from draw_path import draw_path

# 全局路径配置（将在 main 中初始化）
data_source = None
output_dir = None
output_image_path = None
output_path = None
input_path = None
# 使用的经纬度范围（与 FuXi 模型一致）
lat_min = None
lat_max = None
lon_min = None
lon_max = None
# 当前处理的台风 CSV 文件路径（用于提取基准时间）
current_csv = None
from datetime import timedelta

def run_fuxi(model_type, data_source, run_times, base_datetime, output_image_path='output_png'):
    """
    调用 FuXi 模型进行预报。  
    FuXi 模型初始场文件由 download_and_make_era5.py 生成，存放在 input_path 目录下。  
    如果预报步数超过20步，则分段调用 FuXi 模型（最多20步一段）。  
    FuXi 输出文件命名规则为 "{forecast_hour+6:03d}.nc"  
    返回一个 dict，将预报时次（小时）映射到对应 FuXi 输出文件路径。
    """
    input_file = os.path.join(input_path, "input.nc")
    print("调用 FuXi 模型生成预报数据...")
    # 分段设置步数（每步6小时）
    if run_times <= 20:
        steps1, steps2, steps3 = run_times, 0, 0
    elif run_times <= 40:
        steps1, steps2, steps3 = 20, run_times - 20, 0
    else:
        steps1, steps2, steps3 = 20, 20, run_times - 40

    cmd_fuxi = [
        "python", r"FuXi-main/fuxi.py",
        "--model", MODEL_DIR,
        "--input", input_file,
        "--num_steps", str(steps1), str(steps2), str(steps3),
        "--save_dir", output_path
    ]
    print(f"通过命令行调用 FuXi 模型进行预报，预报步长为 {run_times}（共 {run_times * 6} 小时）...")
    subprocess.check_call(cmd_fuxi)

    files_dict = {}
    # 不再调用绘图，仅构建输出文件字典
    for i in range(run_times):
        forecast_hour = (i + 1) * 6
        file_name = os.path.join(output_path, f"{forecast_hour+6:03d}.nc")
        print(f"预报时次 {forecast_hour:02d}h 对应输出文件: {file_name} ...")
        files_dict[forecast_hour] = file_name
    print("FuXi 模型运行完毕！")
    print("生成文件字典:", files_dict)
    return files_dict

def download_ecmwf_forecast(start_time):
    """
    使用 CSV 中的基准时刻下载 ECMWF 预报数据（每6小时一时次）
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
    从台风 CSV 中提取起始时刻，确保时次为 0 或 12
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
    两步链式搜索台风中心：先以初始中心搜索，再以第一次搜索结果进行二次搜索
    """
    first_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, init_lat, init_lon, 2)
    second_search = get_min_pressure_within_range(data_array, lat_arr, lon_arr, first_search['Lat'], first_search['Lon'], 2)
    return second_search

def extract_fuxi_forecast(iso_time, best_lat, best_lon, base_time, data_source, prev_center=None, files_dict=None):
    """
    从 FuXi 模型预报输出（nc 文件）中提取最低气压中心  
    采用两步链式搜索方法，与 ECMWF 保持一致。  
    假设 FuXi 文件命名为 "{forecast_hour+6:03d}.nc"，文件中包含变量 "__xarray_dataarray_variable__" 和 "level" 坐标，
    并在 "level" 坐标中包含 "MSL" 层次。
    """
    forecast_hour = int((iso_time - base_time).total_seconds() // 3600)
    if files_dict is not None and forecast_hour in files_dict:
        file_name = files_dict[forecast_hour]
    else:
        file_name = os.path.join(output_path, f"{forecast_hour+6:03d}.nc")
    print(file_name)
    if not os.path.exists(file_name):
        print(f"未找到 FuXi 模型预报文件: {file_name}")
        return {'MinPressure': np.nan, 'Lat': best_lat, 'Lon': best_lon}
    
    ds = xr.open_dataset(file_name)
    pressure_var = "__xarray_dataarray_variable__"
    msl_indices = np.where(ds['level'].values == 'MSL')[0]
    if msl_indices.size == 0:
        print("未找到 MSL 层")
        ds.close()
        return {'MinPressure': np.nan, 'Lat': best_lat, 'Lon': best_lon}
    msl_index = msl_indices[0]
    pressure_data = ds[pressure_var][0, 0, msl_index, :, :].values / 100
    print(f"FuXi {iso_time} pressure range: {pressure_data.min()} ~ {pressure_data.max()}")
    lat_arr = ds['lat'].values
    lon_arr = ds['lon'].values
    ds.close()
    
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    result = find_typhoon_center(pressure_data, lat_arr, lon_arr, init_lat, init_lon)
    print(f"FuXi {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=None):
    """
    从 ECMWF GRIB 文件中提取最低气压中心  
    链式搜索机制与 FuXi 部分保持一致
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
    pressure_data = msl.values / 100
    ds.close()
    
    init_lat = prev_center[0] if prev_center is not None else best_lat
    init_lon = prev_center[1] if prev_center is not None else best_lon
    result = find_typhoon_center(pressure_data, lats, lons, init_lat, init_lon)
    print(f"ECMWF {iso_time} 最低气压: {result['MinPressure']:.2f} hPa, 对应经纬度: {result['Lat']:.2f}, {result['Lon']:.2f}")
    return result

def merge_data_to_csv(input_csv, output_csv, files_dict=None):
    """
    读取原始台风 CSV，提取每个时次 FuXi 与 ECMWF 的最低气压中心  
    并将结果合并保存到新的 CSV 文件中。  
    files_dict 用于 FuXi 数据文件查找。
    """
    df = pd.read_csv(input_csv)
    df.sort_values(["NAME", "ISO_TIME"], inplace=True)
    base_time = get_start_time(df)
    
    fuxi_results = {}
    ecmwf_results = {}
    
    for name, group in df.groupby("NAME"):
        prev_center_fuxi = None
        prev_center_ecmwf = None
        for idx, row in group.iterrows():
            iso_time = pd.to_datetime(row['ISO_TIME'])
            best_lat, best_lon = row['LAT'], row['LON']
            res_fuxi = extract_fuxi_forecast(iso_time, best_lat, best_lon, base_time, data_source,
                                             prev_center=prev_center_fuxi, files_dict=files_dict)
            fuxi_results[idx] = res_fuxi
            if not np.isnan(res_fuxi['Lat']) and not np.isnan(res_fuxi['Lon']):
                prev_center_fuxi = (res_fuxi['Lat'], res_fuxi['Lon'])
            
            res_ecmwf = extract_ecmwf_forecast(iso_time, best_lat, best_lon, prev_center=prev_center_ecmwf)
            ecmwf_results[idx] = res_ecmwf
            if not np.isnan(res_ecmwf['Lat']) and not np.isnan(res_ecmwf['Lon']):
                prev_center_ecmwf = (res_ecmwf['Lat'], res_ecmwf['Lon'])
    
    df['FuXiMinPressure'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('MinPressure', np.nan))
    df['FuXiMinLat'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('Lat', np.nan))
    df['FuXiMinLon'] = df.index.map(lambda i: fuxi_results.get(i, {}).get('Lon', np.nan))
    df['ECMWFMinPressure'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('MinPressure', np.nan))
    df['ECMWFMinLat'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lat', np.nan))
    df['ECMWFMinLon'] = df.index.map(lambda i: ecmwf_results.get(i, {}).get('Lon', np.nan))
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"合并后的数据已保存到: {output_csv}")

def process_workflow(csv_file):
    global current_csv
    current_csv = csv_file
    df_typhoon = pd.read_csv(csv_file)
    base_time = get_start_time(df_typhoon)
    print(f"处理地形影响台风文件 {csv_file}，基准时刻: {base_time}")
    
    # 通过命令行调用 download_and_make_era5.py 生成初始场数据 input.nc
    input_file = os.path.join(input_path, "input.nc")
    # 修改：传入两个时间参数，以逗号分隔，第一个为基准时间，第二个为基准时间提前6小时
    times_arg = f"{base_time.strftime('%Y%m%d-%H')},{(base_time - timedelta(hours=6)).strftime('%Y%m%d-%H')}"
    cmd_download = [
        "python", r"FuXi-main/download_and_make_era5.py",
        "--times", times_arg,
        "--data_dir", output_dir,
        "--output", input_file
    ]
    print("调用 download_and_make_era5.py 生成初始场数据...")
    subprocess.check_call(cmd_download)
    
    # 计算预报步长：以 CSV 中最后时刻与基准时刻时间差，每6小时为一时次（至少1）
    end_time = pd.to_datetime(df_typhoon['ISO_TIME'].iloc[-1])
    total_hours = (end_time - base_time).total_seconds() / 3600.0
    num_steps = int(total_hours // 6) or 1
    
    # 调用 FuXi 模型进行预报
    files_dict = run_fuxi(6, data_source, run_times=num_steps, 
                          base_datetime=base_time.strftime("%Y%m%d%H"),
                          output_image_path=output_image_path)
    
    # 下载 ECMWF 数据
    print("下载地形影响案例的 ECMWF 预报数据...")
    download_ecmwf_forecast(base_time)
    
    # 合并 FuXi 与 ECMWF 提取结果到 CSV
    base_name = os.path.basename(csv_file)
    output_csv = os.path.join(NEW_DIR, "new_" + base_name)
    merge_data_to_csv(csv_file, output_csv, files_dict=files_dict)

def process_all_csv():
    """遍历 topo-influ-csv 目录处理所有 CSV 文件"""
    TY_PATH = os.path.join("topo-influ-csv")
    for file in os.listdir(TY_PATH):
        if file.endswith(".csv") and not file.startswith("new_"):
            csv_file = os.path.join(TY_PATH, file)
            process_workflow(csv_file)

def main():
    """主函数：初始化全局路径及地理范围参数，并启动处理流程"""
    global data_source, output_dir, output_image_path, output_path, input_path, MODEL_DIR, NEW_DIR
    data_source = 'ERA5'  # 若需要可改为 GFS
    output_dir = 'raw_data'  # 原始数据存储目录
    output_image_path = "output_png"  # 现已不再用于绘图
    output_path = os.path.join("output_data", "fuxi")
    input_path = "input"
    MODEL_DIR = "model"  # FuXi 模型存放目录
    NEW_DIR = os.path.join("topo-influ-csv", "new_")  # 输出 CSV 存储目录
    
    # 初始化地理范围（根据实际区域进行调整）
    lon_min = 115   # 经度下限
    lon_max = 150   # 经度上限
    lat_min = 15    # 纬度下限
    lat_max = 30    # 纬度上限
    globals().update(locals())

    # 创建必要目录
    os.makedirs(NEW_DIR, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    process_all_csv()

if __name__ == "__main__":
    main()