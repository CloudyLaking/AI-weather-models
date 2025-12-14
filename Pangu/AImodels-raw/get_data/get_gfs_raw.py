import numpy
import requests
import os
import xarray as xr
from tqdm import tqdm

# 下载GFS数据
def download(year, month, day, hour, download_dir='raw_data'):
    #url=f'https://data.rda.ucar.edu/d084003/{year}/{year}{month}{day}/gfs.0p25b.{year}{month}{day}{hour}.f000.grib2'
    #url = f'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001/{year}/{year}{month}{day}/gfs.0p25.{year}{month}{day}{hour}.f000.grib2'
    url = 'https://data.rda.ucar.edu/d083002/grib2/{year}/{year}.{month}/fnl_{year}{month}{day}_{hour}_00.grib2'

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    filename = os.path.join(download_dir, f'fnl_{year}{month}{day}_{hour}_00.grib2')
    if not os.path.exists(filename):
        print(f'Downloading {filename}...')
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kilobyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")
        else:
            print(f'{filename} downloaded.')
    else:
        print(f'{filename} already exists.')
    return filename

# 使用cfgrib处理GRIB文件
def data_process(filename, output_dir='input'):
    # 处理地面变量
    surface_vars = {}
    
    # 读取t2m
    ds_t2m = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround'})
    surface_vars['t2m'] = ds_t2m['t2m'].values
    
    # 读取mslet
    ds_mslet = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'})
    surface_vars['mslet'] = ds_mslet['mslet'].values
    
    # 读取v10和u10
    ds_v10_u10 = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'planetaryBoundaryLayer'})
    surface_vars['v10'] = ds_v10_u10['v'].values
    surface_vars['u10'] = ds_v10_u10['u'].values
    
    # 将地面变量组合成一个数组
    data_surface = numpy.stack([surface_vars['mslet'], surface_vars['u10'], surface_vars['v10'], surface_vars['t2m']])

    pressure_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存地面变量
    output_surface_filename = os.path.join(output_dir, 'input_surface.npy')
    numpy.save(output_surface_filename, data_surface)
    print(f'Surface data saved as {output_surface_filename}')
    
    return data_surface

# 打印GRIB文件信息
def print_grib_info(filename):
    ds = xr.open_dataset(filename, engine='cfgrib')
    print("GRIB File Variables:")
    print(ds)

# 主函数
if __name__ == '__main__':
    inputs='2021041700'
    year = inputs[:4]
    month = inputs[4:6]
    day = inputs[6:8]
    hour = inputs[8:]
    filename = download(year, month, day, hour)
    data_surface = data_process(filename)
    print(f'Surface data shape: {data_surface.shape}')