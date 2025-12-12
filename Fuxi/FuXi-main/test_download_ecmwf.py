#!/usr/bin/env python
import xarray as xr
import numpy as np
import pandas as pd

def main():
    grib_path = "raw_input/ecmwf_forecast.grib"
    try:
        ds = xr.open_dataset(grib_path, engine="cfgrib", decode_timedelta=False)
    except Exception as e:
        print("读取 GRIB 文件失败:", e)
        return

    # 检查是否含有海平面气压数据
    if "msl" not in ds:
        print("GRIB 文件中不存在 'msl' 变量")
        return

    msl = ds["msl"]
    
    # 输出数据集基础信息
    print("数据集基本信息:")
    print(ds)
    
    # 使用 valid_time 作为各个预报时效
    if "valid_time" in ds.coords:
        times = pd.to_datetime(ds["valid_time"].values, errors="coerce")
        for idx, t in enumerate(times):
            try:
                # 选择对应时次的数据
                ds_sel = msl.isel(step=idx)
                data = ds_sel.values
                print(f"预报时效: {t}")
                print(f"最小气压: {np.nanmin(data):.2f} hPa, 最大气压: {np.nanmax(data):.2f} hPa, 平均气压: {np.nanmean(data):.2f} hPa")
            except Exception as e:
                print(f"处理时次 {t} 数据失败: {e}")
    else:
        print("无 valid_time 坐标，直接输出气压数据统计:")
        data = msl.values
        print(f"最小气压: {np.nanmin(data):.2f} hPa, 最大气压: {np.nanmax(data):.2f} hPa, 平均气压: {np.nanmean(data):.2f} hPa")
    
if __name__ == "__main__":
    main()