import os

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

import cfgrib

def make_era5(pl, sfc, tp):

    # pl = xr.open_dataset(pl_filename)

    # sfc = xr.open_dataset(sfc_filename)

    # tp = xr.open_dataarray(tp_filename).fillna(0) * 1000 #单位为mm
    tp = tp.clip(min=0, max=1000)
    sfc['tp'] = tp # 注意这个地方有问题的原因在于两个dataarray中coords的time不一样！！！
    # sfc['tp'].values = np.zeros(sfc['tp'].values.shape, dtype = 'float32')
    # print(np.isnan(sfc['tp']).sum())

    pl_names = ['z', 't', 'u', 'v', 'r']
    sfc_names = ['t2m', 'u10', 'v10', 'msl', 'tp']
    levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    channel = [f'{n.upper()}{l}' for n in pl_names for l in levels]
    channel +=[n.upper() for n in sfc_names]

    ds = []
    for name in pl_names + sfc_names:
        if name in ['z', 't', 'u', 'v', 'r']:
            v = pl[name]
            v = v.reindex(level=levels)

        if name in ['t2m', 'u10', 'v10', 'msl', 'tp']:
            v = sfc[name]
            level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
            v = v.expand_dims({'level': level}, axis=1)             

        if np.isnan(v).sum() > 0:
            # print(np.isnan(v).sum())
            print(f"{name} has nan value")
            raise ValueError

        v.name = "data"
        v.attrs = {}                
        # print(f"{name}: {v.shape}, {v.min().values} ~ {v.max().values}")
        ds.append(v) # 输出格式与v相同，即dataarray
     
    ds = xr.concat(ds, 'level')
    ds = ds.assign_coords(level=channel)
    ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    ds = ds.astype(np.float32)
    return ds



def load_from_era5(filenames_01, filenames_02):
    pl_filename_01, sfc_filename_01, tp_filename_01 = filenames_01
    pl_filename_02, sfc_filename_02, tp_filename_02 = filenames_02
    ds1 = make_era5(pl_filename_01, sfc_filename_01, tp_filename_01)
    ds2 = make_era5(pl_filename_02, sfc_filename_02, tp_filename_02)
    ds = xr.concat([ds1, ds2], 'time')
    return ds




def get_era5_filename(init_time_02):
    delta_hr = 6 #需要读取起报时刻前6小时的数据
    init_time_01 = init_time_02 - timedelta(hours = delta_hr)
    
    ERA5_dir = '../../ERA5_input_data/'
    
    pl_filename_01 = init_time_01.strftime(ERA5_dir + 'ERA5_upper/ERA5_yr%Y_mm%m_dd%d_%H00_upper.nc')
    sfc_filename_01 = init_time_01.strftime(ERA5_dir + 'ERA5_surface/ERA5_yr%Y_mm%m_dd%d_%H00_surface.nc')
    tp_filename_01 = init_time_01.strftime(ERA5_dir + 'ERA5_precipitation/ERA5_yr%Y_mm%m_dd%d_%H00_precipitation.nc')
    filenames_01 = (pl_filename_01, sfc_filename_01, tp_filename_01)
    
    pl_filename_02 = init_time_02.strftime(ERA5_dir+'ERA5_upper/ERA5_yr%Y_mm%m_dd%d_%H00_upper.nc')
    sfc_filename_02 = init_time_02.strftime(ERA5_dir+'ERA5_surface/ERA5_yr%Y_mm%m_dd%d_%H00_surface.nc')
    tp_filename_02 = init_time_02.strftime(ERA5_dir+'ERA5_precipitation/ERA5_yr%Y_mm%m_dd%d_%H00_precipitation.nc')
    filenames_02 = (pl_filename_02, sfc_filename_02, tp_filename_02) 
    return filenames_01, filenames_02 

if __name__ == "__main__": 
    # import argparse
    # parser = argparse.ArgumentParser(description='fuxi_foreward_ctrl')
    # parser.add_argument('--init_time', required=True, type=str)
    # parser.add_argument('--tc_lat', required=True, type=float)
    # parser.add_argument('--tc_lon', required=True, type=float)
    # args = parser.parse_args()
    # init_time_str, tc_lat, tc_lon = \
    #     args.init_time, args.tc_lat, args.tc_lon
    
    # init_time = datetime.strptime(init_time_str, '%Y-%m-%d_%H:%M:%S')
        
    # 必须去linux系统上运行！！！！windows会报错oserror!!!
    all_need_times = []
    start_time = datetime(2020,10,5,12,0,0)#init_time# 
    d = start_time - timedelta(hours = 6)
    while d <= start_time + timedelta(days = 5):#datetime(2020,10,10,12,0,0):
        all_need_times.append(d)
        d = d + timedelta(hours = 6)
    
    # # 高层数据
    # # upper_dir = './ERA5_upper/'
    # # if os.path.exists(upper_dir) == False:
    # #     os.makedirs(upper_dir)
    pl_all = xr.open_dataset('ERA5_Chanhom_upper.grib', engine='cfgrib')
    
    # 地面数据
    # grib的key可以通过grib_ls ERA5_Chanhom_surface.grib获取
    # sfc_dir = './ERA5_surface/'
    # if os.path.exists(sfc_dir) == False:
    #     os.makedirs(sfc_dir)
    # grib的key可以通过grib_ls ERA5_Chanhom_surface.grib获取
    sfc_all = xr.open_dataset('ERA5_Chanhom_surface.grib', engine='cfgrib', 
                               backend_kwargs={'filter_by_keys':{'typeOfLevel':'surface',
                                                                 'stepRange':'0',
                                                                 }})#'surface':0
    # print(sfc_all)
    # sfc_all.to_netcdf('test.nc')
    # # 降水数据
    # # pre_dir = './ERA5_precipitation/'
    # # if os.path.exists(pre_dir) == False:
    # #     os.makedirs(pre_dir)
    tp_all = xr.open_dataarray('ERA5_Chanhom_precipitation.nc').fillna(0)
    tp_all = tp_all.rename({'valid_time': 'time'})
    
    # fuxi格式数据
    fuxi_std_dir = './ERA5_fuxi_std/'
    if os.path.exists(fuxi_std_dir) == False:
        os.makedirs(fuxi_std_dir)
    
    # wrf_input格式数据
    wrf_input_dir = './WRF_input_std/'
    if os.path.exists(wrf_input_dir) == False:
        os.makedirs(wrf_input_dir)
    
    
    ds_list = []
    # 构造fuxi格式nc数据
    for ii in range(len(all_need_times)):
        need_time = all_need_times[ii]
        yr = '%04d'%need_time.year
        mo = '%02d'%need_time.month
        da = '%02d'%need_time.day
        hr = '%02d'%need_time.hour 
        print(need_time)
    
        # 选取高层数据
        # , step=0, number = 1, vaild_time = np.datetime64(need_time).astype('datetime64[ns]')
        pl = pl_all.sel(time = [np.datetime64(need_time).astype('datetime64[ns]')])
        # 需要对高层数据进行压缩：
        # comp = dict(zlib=True, complevel=5)
        # encoding = {var: comp for var in pl.data_vars}
        pl = pl.rename({'isobaricInhPa': r'level'})
        pl = pl.drop_vars('number')
        pl = pl.drop_vars('step')
        pl = pl.drop_vars('valid_time')
        pl = pl.assign_coords(step=('time', [(ii-1)*6]))
        pl = pl.swap_dims({'time':'step'})
        pl = pl.drop_vars('time')
        pl = pl.expand_dims('time')
        pl = pl.assign_coords(time=[np.datetime64(all_need_times[1])])# 先分配数轴coords，再分配维度dim
        
        # pl.to_netcdf(upper_dir + 'ERA5_yr'+yr+'_mm'+mo+'_dd'+da+'_'+hr+'00'+'_upper.nc')#, encoding=encoding
        # pl.to_netcdf(r'D:\\test.nc')
        # 选取地面数据
        sfc = sfc_all.sel(time = [np.datetime64(need_time).astype('datetime64[ns]')])
        sfc = sfc.drop_vars('number')
        sfc = sfc.drop_vars('step')
        sfc = sfc.drop_vars('valid_time')
        sfc = sfc.drop_vars('surface')
        sfc = sfc.assign_coords(step=('time', [(ii-1)*6]))
        sfc = sfc.swap_dims({'time':'step'})
        sfc = sfc.drop_vars('time')
        sfc = sfc.expand_dims('time')
        sfc = sfc.assign_coords(time=[np.datetime64(all_need_times[1])])
        # # sfc = sfc.expand_dims('step')
        # sfc.to_netcdf(sfc_dir + 'ERA5_yr'+yr+'_mm'+mo+'_dd'+da+'_'+hr+'00'+'_surface.nc')
        # 读入降水数据并处理
        
        dt_0 = np.datetime64(need_time) - np.timedelta64(5, 'h')
        dt_1 = np.datetime64(need_time) + np.timedelta64(1, 'h')
        dts = np.arange(dt_0, dt_1, np.timedelta64(1, 'h')).astype('datetime64[ns]')
        sum_end_dt = dts[-1]
        tp = tp_all.sel(time = dts).sum(dim='time')
        tp = tp.drop_vars('number')
        # tp = tp.drop_vars('time')
        tp = tp.expand_dims('time')
        tp = tp.assign_coords(time = [np.datetime64(all_need_times[1])])
        tp = tp.expand_dims('step')
        tp = tp.assign_coords(step = [(ii-1)*6])
        
        # tp_new.to_netcdf(pre_dir + 'ERA5_yr'+yr+'_mm'+mo+'_dd'+da+'_'+hr+'00'+'_precipitation.nc')
        ds = make_era5(pl, sfc, tp)
        ds.to_netcdf(f'./ERA5_fuxi_std/{(ii-1)*6:03d}.nc')
        ds_list.append(ds)
            
    
    init_era5_input = xr.concat([ds_list[0].sel(step = -6), ds_list[1].sel(step = 0)], 'time')
    init_era5_input = init_era5_input.drop_vars('step')
    init_era5_input.to_netcdf('./fuxi_input.nc')
    
    # # # 读取初始条件作为输入
    # # filenames_01, filenames_02 = get_era5_filename(start_time)
    # # init_era5_input = load_from_era5(filenames_01, filenames_02) 
    # # init_era5_input.to_netcdf('../fuxi_input.nc')
    
    
    # data_in_srf = pg.open('ERA5_Chanhom_surface.grib')
    # for need_time in all_need_times:
    #     # 注意每次从头读取都需要返回文件头
    #     data_in_srf.seek(0)
    #     filename_srf = need_time.strftime('ERA5_yr%Y_mm%m_dd%d_%H00_surface.grib')
    #     grbout = open(wrf_input_dir + filename_srf,'wb')
    #     for grb in data_in_srf:
    #         date_now_grib = grb.analDate + timedelta(hours = grb.endStep)
    #         if date_now_grib == need_time:
    #             msg = grb.tostring()
    #             grbout.write(msg)    
    #     grbout.close()
    
    
    # # # 使用pygrib读取高层数据
    # data_in_pl = pg.open('ERA5_Chanhom_upper.grib')#_old
    
    # for need_time in all_need_times:    
    #     data_in_pl.seek(0)
    #     # date_now_grib, time_now_grib = grib_timestyle(need_time)
    #     filename_pl = need_time.strftime('ERA5_yr%Y_mm%m_dd%d_%H00_upper.grib')
    #     grbout = open(wrf_input_dir + filename_pl,'wb')
    #     for grb in data_in_pl:
    #         date_now_grib = grb.analDate + timedelta(hours = grb.endStep)
    #         if date_now_grib == need_time:
    #             msg = grb.tostring()
    #             grbout.write(msg)  
    #     grbout.close()
        
    # data_in_pl.close()


    # # 台风定位
    # io_data_dir = [fuxi_std_dir]
    # tc_lat = 23.
    # tc_lon = 139.3
    # tc_tracker(io_data_dir, tc_lat, tc_lon, start_time)
    
    
    # # 画500hPa和850hPa图
    # levels = [500, 850]
    # io_dirs = [fuxi_std_dir]
    # typhoon_pic_multifig_by_time(levels, io_dirs, start_time)
    
    
    # # 画500hPa和850hPa图
    # levels = [500, 850]
    # io_dirs = [fuxi_std_dir]
    # typhoon_pic_multifig_by_time_daily(levels, io_dirs, start_time)
