import os
import xarray as xr
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from tqdm import tqdm

def get_wind_data(output_dir, datetime):
    # 下载的url
    base_url = 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001'
    year = datetime[:4]
    date = datetime[:8]
    filename = f'gfs.0p25.{datetime}.f000.grib2'
    url = f'{base_url}/{year}/{date}/{filename}'

    grib_filename = f'gfs_{datetime}.grib2'
    grib_file_path = os.path.join(output_dir, grib_filename)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 下载文件并显示进度条
    def download_with_progress_bar(url, target):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        response = session.get(url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(target, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            print("ERROR, something went wrong")

    # 检查文件是否存在
    if not os.path.exists(grib_file_path):
        download_with_progress_bar(url, grib_file_path)
    else:
        print(f"File {grib_file_path} already exists, skipping download.")

    # 读取grib文件并提取风速数据
    def extract_wind_data(grib_file_path):
        ds_surface = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'sigma'})
        
        # 打印所有变量名称
        print("Variables in the surface level:")
        for var in ds_surface.variables:
            print(var)
        
        if 'u' in ds_surface and 'v' in ds_surface:
            u10 = ds_surface['u'].values
            v10 = ds_surface['v'].values
            wind_speed = np.sqrt(u10**2 + v10**2)
            
            max_wind_speed = np.max(wind_speed)
            mean_wind_speed = np.mean(wind_speed)
            
            print(f"Max Wind Speed: {max_wind_speed:.2f} m/s")
            print(f"Mean Wind Speed: {mean_wind_speed:.2f} m/s")
        else:
            print(f"Wind data not found in {grib_file_path}")
        if 'VRATE' in ds_surface :
            vrate = ds_surface['VRATE'].values
            
            max_wind_speed = np.max(vrate)
            mean_wind_speed = np.mean(vrate)
            
            print(f"Max Wind Speed: {max_wind_speed:.2f} m/s")
            print(f"Mean Wind Speed: {mean_wind_speed:.2f} m/s")
        else:
            print(f"Wind data not found in {grib_file_path}")
        if 'gust' in ds_surface :
            vrate = ds_surface['gust'].values
            
            max_wind_speed = np.max(vrate)
            mean_wind_speed = np.mean(vrate)
            
            print(f"Max Wind Speed: {max_wind_speed:.2f} m/s")
            print(f"Mean Wind Speed: {mean_wind_speed:.2f} m/s")
        else:
            print(f"Wind data not found in {grib_file_path}")

    extract_wind_data(grib_file_path)

if __name__ == '__main__':
    output_dir = 'input'
    datetime = '2021041700'
    get_wind_data(output_dir, datetime)