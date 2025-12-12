import os
import requests
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    chunk_size = 8192
    with open(dest_path, 'wb') as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)
        ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print("下载完成。")

def construct_gfs_url(time_str):
    # time_str 格式为 "YYYYMMDDHH", 如 "2023010100"
    year = time_str[:4]
    yyyymmdd = time_str[:8]
    url = f"https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/{year}/{yyyymmdd}/gfs.0p25.{time_str}.f000.grib2"
    return url

def make_gfs(src_name):
    assert os.path.exists(src_name)
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ['gh', 't', 'u', 'v', 'r']
    # 原 sf_names 列表只用于映射，实际通过 shortName 读取
    sf_names = ['2t', '10u', '10v', 'mslet']
    name_map = {'2t': 't2m', '10u': 'u10', '10v': 'v10', 'mslet': 'msl'}
    
    try:
        ds_pl = xr.open_dataset(src_name, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}
        })
    except Exception as e:
        print(f"读取大气层数据失败: {e}")
        return

    inputs = []
    level_names = []
    init_time = None
    lat = None
    lon = None

    # 处理大气层变量
    for name in pl_names:
        output_name = 'z' if name == 'gh' else name
        for lev in levels:
            try:
                da = ds_pl[name].sel(isobaricInhPa=lev)
            except Exception as e:
                print(f"大气层变量 {name} 在 {lev}hPa 不存在: {e}")
                ds_pl.close()
                return
            if init_time is None:
                try:
                    init_time = pd.to_datetime(da.valid_time.values)
                except:
                    init_time = pd.Timestamp.now()
            if lat is None:
                lat = da.latitude
            if lon is None:
                lon = da.longitude
            data = da.data
            if name == 'gh':
                data = data * 9.8  # 转换为位势
            inputs.append(data)
            level_names.append(f"{output_name}{lev}")
            print(f"{name} at {lev}: shape {data.shape}")
    ds_pl.close()

    # 处理地面变量，使用 shortName 读取
    for name in sf_names:
        shortName = name_map[name]
        try:
            ds_sf = xr.open_dataset(src_name, engine='cfgrib', backend_kwargs={
                'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'shortName': shortName}
            })
        except Exception as e:
            print(f"地面变量 {name} 不存在: {e}")
            return
        if lat is None:
            lat = ds_sf.latitude
        if lon is None:
            lon = ds_sf.longitude
        try:
            da = ds_sf[shortName]
        except Exception as e:
            print(f"无法读取地面变量 {name} 对应的 {shortName}: {e}")
            ds_sf.close()
            return
        inputs.append(da.data)
        level_names.append(shortName)
        print(f"{name} -> {shortName}: shape {da.data.shape}")
        ds_sf.close()
    
    # 处理累积降水量 tp
    try:
        ds_tp = xr.open_dataset(src_name, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'heightAboveGround', 'shortName': 'tp'}
        })
        da_tp = ds_tp['tp']
        inputs.append(da_tp.data)
        level_names.append("tp")
        print(f"tp: shape {da_tp.data.shape}")
        ds_tp.close()
    except Exception as e:
        print(f"无法读取 tp 数据, 构造全 0 数组。错误: {e}")
        if len(inputs) > 0:
            shape = inputs[-1].shape
            tp = np.zeros(shape, dtype=inputs[-1].dtype)
            inputs.append(tp)
            level_names.append("tp")
            print(f"构造 tp: shape {tp.shape}")
        else:
            print("未能获取 tp 数据")
            return

    inputs = np.stack(inputs)
    assert inputs.shape[0] == 70, f"输入变量数为 {inputs.shape[0]}，不是 70"
    
    times = [init_time]
    data_array = xr.DataArray(
        data=inputs[None],
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': times, 'level': level_names, 'lat': lat, 'lon': lon},
    )

    if np.isnan(data_array).sum() > 0:
        print("存在 NaN 值")
        return 
    
    return data_array

def main():
    parser = argparse.ArgumentParser(
        description="下载 GFS 初始场 grib 文件并制作 input.nc 文件"
    )
    parser.add_argument("--times", type=str, default="2023010100,2023010112",
                        help="逗号分隔的初始时次列表，格式 YYYYMMDDHH，如 2023010100,2023010112")
    parser.add_argument("--output", type=str, default="input.nc",
                        help="输出文件名，默认为 input.nc")
    args = parser.parse_args()

    folder = "gfs_input"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"创建文件夹: {folder}")

    time_list = [t.strip() for t in args.times.split(",")]
    data_arrays = []
    for t_str in time_list:
        url = construct_gfs_url(t_str)
        hour = t_str[8:10]
        file_name = f"gfs.t{hour}z.0p25.{t_str}.f000.grib2"
        dest_path = os.path.join(folder, file_name)
        if not os.path.exists(dest_path):
            try:
                download_file(url, dest_path)
            except Exception as e:
                print(f"下载 {file_name} 失败: {e}")
                continue
        else:
            print(f"{file_name} 已存在，跳过下载。")
        da = make_gfs(dest_path)
        if da is not None:
            data_arrays.append(da)
        else:
            print(f"处理 {file_name} 失败。")
    
    if len(data_arrays) > 0:
        ds_all = xr.concat(data_arrays, dim="time")
        ds_all = ds_all.assign_coords(time=ds_all.time.astype(np.datetime64))
        ds_all.to_netcdf(args.output)
        print(f"生成 {args.output} 成功！")
    else:
        print("未能生成有效数据，程序终止。")

if __name__ == "__main__":
    main()