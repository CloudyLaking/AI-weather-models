#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import argparse

def download_pl(init_time, data_dir):
    """
    下载 ERA5 压力层数据：变量包括 geopotential, temperature, u/v wind, relative humidity
    """
    init_dt = pd.to_datetime(init_time, format="%Y%m%d-%H")
    out_file = os.path.join(data_dir, init_dt.strftime('P%Y%m%d%H.nc'))
    if os.path.exists(out_file):
        print(f"压力层数据 {out_file} 已存在")
        return
    c = cdsapi.Client()
    params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['geopotential', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'relative_humidity'],
        'pressure_level': [str(p) for p in [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]],
        'year': init_dt.strftime('%Y'),
        'month': init_dt.strftime('%m'),
        'day': init_dt.strftime('%d'),
        'time': init_dt.strftime('%H:00'),
    }
    print(f"正在下载压力层数据到 {out_file} ...")
    c.retrieve('reanalysis-era5-pressure-levels', params, out_file)
    print("下载完成")

def download_sfc(init_time, data_dir):
    """
    下载 ERA5 近地面数据（不含降水）：变量包括2m_temperature, 10m_u/v wind, mean_sea_level_pressure
    """
    init_dt = pd.to_datetime(init_time, format="%Y%m%d-%H")
    out_file = os.path.join(data_dir, init_dt.strftime('S%Y%m%d%H.nc'))
    if os.path.exists(out_file):
        print(f"近地面数据 {out_file} 已存在")
        return
    c = cdsapi.Client()
    params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure'],
        'year': init_dt.strftime('%Y'),
        'month': init_dt.strftime('%m'),
        'day': init_dt.strftime('%d'),
        'time': init_dt.strftime('%H:00'),
    }
    print(f"正在下载近地面数据到 {out_file} ...")
    c.retrieve('reanalysis-era5-single-levels', params, out_file)
    print("下载完成")

def download_tp(init_time, data_dir):
    """
    下载 ERA5 降水数据（总降水）：自动处理跨日情况
    """
    init_dt = pd.to_datetime(init_time, format="%Y%m%d-%H")
    out_file = os.path.join(data_dir, init_dt.strftime('R%Y%m%d.nc'))
    if os.path.exists(out_file):
        print(f"降水数据 {out_file} 已存在")
        return

    c = cdsapi.Client()
    start_dt = init_dt - pd.Timedelta(hours=6)

    def download_segment(dt, times, suffix):
        params = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['total_precipitation'],
            'year': dt.strftime('%Y'),
            'month': dt.strftime('%m'),
            'day': dt.strftime('%d'),
            'time': times,
        }
        file_path = os.path.join(data_dir, dt.strftime(f'R%Y%m%d{suffix}.nc'))
        if not os.path.exists(file_path):
            print(f"正在下载降水数据到 {file_path} ...")
            c.retrieve('reanalysis-era5-single-levels', params, file_path)
            print("下载完成")
        return file_path

    if start_dt.day == init_dt.day:
        # 未跨日时直接下载
        times = [f"{h:02d}:00" for h in range(start_dt.hour, init_dt.hour)]
        params = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['total_precipitation'],
            'year': init_dt.strftime('%Y'),
            'month': init_dt.strftime('%m'),
            'day': init_dt.strftime('%d'),
            'time': times,
        }
        print(f"正在下载降水数据到 {out_file} ...")
        c.retrieve('reanalysis-era5-single-levels', params, out_file)
        print("下载完成")
    else:
        # 跨日情况，分别下载前一天和当天的部分数据后合并
        times_prev = [f"{h:02d}:00" for h in range(start_dt.hour, 24)]
        times_curr = [f"{h:02d}:00" for h in range(0, init_dt.hour)]
        prev_file = download_segment(start_dt, times_prev, '_part')
        da_prev = xr.open_dataarray(prev_file).fillna(0)
        if times_curr:
            curr_file = download_segment(init_dt, times_curr, '_part')
            da_curr = xr.open_dataarray(curr_file).fillna(0)
            dim_time = "time" if "time" in da_prev.dims else "valid_time"
            da_sum = (xr.concat([da_prev, da_curr], dim=dim_time).sum(dim=dim_time) * 1000).clip(min=0, max=1000)
        else:
            # 当当前段没有数据时，仅使用前天段数据
            dim_time = "time" if "time" in da_prev.dims else "valid_time"
            da_sum = (da_prev.sum(dim=dim_time) * 1000).clip(min=0, max=1000)
        if "time" not in da_sum.dims:
            da_sum = da_sum.expand_dims({"time": [init_dt]})
        da_sum.to_netcdf(out_file)
        print(f"跨日降水数据合并并保存到 {out_file}")

def download_era5_data(init_time, data_dir):
    """
    根据所需时次下载ERA5数据文件
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    download_pl(init_time, data_dir)
    download_sfc(init_time, data_dir)
    download_tp(init_time, data_dir)

def make_era5(init_time, data_dir):
    init_time = pd.to_datetime(init_time, format="%Y%m%d-%H")
    print(f"处理 {init_time} ...")
    
    # 下载数据（如果尚未下载）
    download_era5_data(init_time.strftime("%Y%m%d-%H"), data_dir)
    
    pl_file = os.path.join(data_dir, init_time.strftime('P%Y%m%d%H.nc'))
    pl = xr.open_dataset(pl_file)
    sfc_file = os.path.join(data_dir, init_time.strftime('S%Y%m%d%H.nc'))
    sfc = xr.open_dataset(sfc_file)

    tp_file = os.path.join(data_dir, init_time.strftime('R%Y%m%d.nc'))
    tp = xr.open_dataarray(tp_file).fillna(0)
    # 对降水数据直接求和得到6小时总降水（单位转换为毫米）
    dim_time = "time" if "time" in tp.dims else "valid_time"
    tp_6h = tp.sum(dim=dim_time) * 1000
    tp_6h = tp_6h.clip(min=0, max=1000)
    # 为确保统一，若 tp_6h 没有 time 维度则扩展为单一的 time
    if "time" not in tp_6h.dims:
        tp_6h = tp_6h.expand_dims({"time": [init_time]})
    sfc['tp'] = tp_6h

    pl_names = ['z', 't', 'u', 'v', 'r']
    sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    # 修改：对于气压层数据，调整为逆序排列，确保数据顺序与命名一致
    # 构造通道名称，气压层逆序排列
    channel = [f'{n.upper()}{l}' for n in pl_names for l in reversed(levels)]
    channel += [n.upper() for n in sfc_names]

    ds_list = []
    for name in pl_names + sfc_names:
        if name in pl_names:
            v = pl[name]
            # 如果存在 "pressure_level" 坐标，则重命名为 "level" 并逆序排列气压层数据
            if "pressure_level" in v.coords:
                v = v.rename({"pressure_level": "level"})
                v = v.isel(level=slice(None, None, -1))
        elif name in sfc_names:
            v = sfc[name]
            # 扩展 level 维度，使近地面变量也具有 level 这一坐标
            level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
            v = v.expand_dims({'level': level}, axis=1)
        else:
            continue

        # 去除可能存在的不一致坐标，如 valid_time
        if "valid_time" in v.coords:
            v = v.drop_vars("valid_time")
        if np.isnan(v).sum() > 0:
            print(f"{name} 存在 nan 值")
            raise ValueError(f"{name} contains NaN")

        v.name = "data"
        v.attrs = {}
        print(f"{name}: {v.shape}, {v.min().values} ~ {v.max().values}")
        ds_list.append(v)
     
    ds = xr.concat(ds_list, dim='level')
    ds = ds.assign_coords(level=channel)
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    ds = ds.astype(np.float32)
    # 如果已有 "time" 维度，则赋予新的时间坐标，否则扩展时间维度
    if "time" in ds.dims:
        ds = ds.assign_coords(time=[init_time])
    else:
        ds = ds.expand_dims({'time': [init_time]})
    return ds

def main():
    parser = argparse.ArgumentParser(description="生成ERA5历史数据input.nc文件（含下载部分）")
    parser.add_argument("--times", type=str, default="20200525-12,20200525-18",
                        help="逗号分隔的初始化时次列表，格式如 20230725-12,20230725-18")
    parser.add_argument("--data_dir", type=str, default="raw_input",
                        help="ERA5数据存放目录")
    parser.add_argument("--output", type=str, default="input.nc",
                        help="输出文件名，默认 input.nc")
    args = parser.parse_args()

    init_times = [t.strip() for t in args.times.split(",") if t.strip()]
    ds_all = []
    for t in init_times:
        ds = make_era5(t, args.data_dir)
        ds_all.append(ds)
    ds_concat = xr.concat(ds_all, dim='time')
    
    # 如果存在 valid_time 坐标，则移除
    if "valid_time" in ds_concat.coords:
        ds_concat = ds_concat.drop_vars("valid_time")
    
    # 将 time 维度置于第一个位置，其余维度顺序保持不变
    dims = list(ds_concat.dims)
    if "time" in dims:
        dims.remove("time")
        new_order = ("time",) + tuple(dims)
        ds_concat = ds_concat.transpose(*new_order)
    
    # 删除第二个维度（大小为1）的维度
    dims_order = list(ds_concat.dims)
    if len(dims_order) > 1:
        second_dim = dims_order[1]
        if ds_concat.sizes[second_dim] == 1:
            ds_concat = ds_concat.squeeze(dim=second_dim)
    
    ds_concat.to_netcdf(args.output)
    print(f"数据保存到 {args.output}")

if __name__ == "__main__":
    main()