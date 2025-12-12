import os
import numpy as np
import pandas as pd
import xarray as xr

def make_gfs(src_name):
    assert os.path.exists(src_name)

    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    pl_names = ['gh', 't', 'u', 'v', 'r']
    sf_names = ['2t', '10u', '10v', 'mslet']
    
    # 读取大气层数据（等压面）
    try:
        ds_pl = xr.open_dataset(src_name, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}
        })
    except Exception as e:
        print(f"读取大气层数据失败: {e}")
        return

    # 读取地面数据（高度面）
    try:
        ds_sf = xr.open_dataset(src_name, engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}
        })
    except Exception as e:
        print(f"读取地面数据失败: {e}")
        return

    inputs = []
    level_names = []
    init_time = None
    lat = None
    lon = None

    # 处理大气层变量
    for name in pl_names:
        # 如果变量为 gh，则输出命名为 z，并乘以重力加速度转换为位势
        output_name = 'z' if name == 'gh' else name
        for lev in levels:
            try:
                # 选择等压面
                da = ds_pl[name].sel(isobaricInhPa=lev)
            except Exception as e:
                print(f"大气层变量 {name} 在 {lev}hPa 不存在: {e}")
                return
            # 初始时间只取一次
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
                data = data * 9.8
            inputs.append(data)
            level_names.append(f"{output_name}{lev}")
            print(f"{name} at {lev}: shape {data.shape}")
    
    # 处理地面变量
    name_map = {'2t': 't2m', '10u': 'u10', '10v': 'v10', 'mslet': 'msl'}
    for name in sf_names:
        out_name = name_map[name]
        try:
            da = ds_sf[name]
        except Exception as e:
            print(f"地面变量 {name} 不存在: {e}")
            return
        if lat is None:
            lat = da.latitude
        if lon is None:
            lon = da.longitude
        inputs.append(da.data)
        level_names.append(out_name)
        print(f"{name} -> {out_name}: shape {da.data.shape}")
    
    # 处理累积降水量 tp：如果数据中存在 tp，则直接读取，否则构造全 0 数组，形状参考最后一个地面变量
    if 'tp' in ds_sf.data_vars:
        da_tp = ds_sf['tp']
        inputs.append(da_tp.data)
        level_names.append("tp")
        print(f"tp: shape {da_tp.data.shape}")
    else:
        # 构造全 0 数组（假定地面变量形状一致）
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
    # 期望变量数 70 (大气层 5*13 + 地面 4 + tp  = 65 + 5)
    assert inputs.shape[0] == 70, f"输入变量数为 {inputs.shape[0]}，不是 70"
    
    # 构造 DataArray, 增加 time 维度
    times = [init_time]
    data_array = xr.DataArray(
        data=inputs[None],  # shape (time, level, lat, lon)
        dims=['time', 'level', 'lat', 'lon'],
        coords={'time': times, 'level': level_names, 'lat': lat, 'lon': lon},
    )

    if np.isnan(data_array).sum() > 0:
        print("存在 NaN 值")
        return 
    
    return data_array

def test_make_gfs():
    d1 = make_gfs('30/gfs.t06z.pgrb2.0p25.f000')
    d2 = make_gfs('30/gfs.t12z.pgrb2.0p25.f000')

    if d1 is not None and d2 is not None:
        ds = xr.concat([d1, d2], 'time')
        ds = ds.assign_coords(time=ds.time.astype(np.datetime64))
        ds.to_netcdf('input.nc')