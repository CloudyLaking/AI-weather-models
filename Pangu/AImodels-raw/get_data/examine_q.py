import os
import xarray as xr
import numpy as np
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_gfs_data(output_dir, datetime):

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

    # 读取grib文件并比较q值
    def compare_q_values(output_dir, grib_filename):
        grib_file_path = os.path.join(output_dir, grib_filename)
        if os.path.exists(grib_file_path):
            ds_upper = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, errors='ignore')
            
            levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            
            temperature = ds_upper['t'].sel(isobaricInhPa=levels).values
            relative_humidity = ds_upper['r'].sel(isobaricInhPa=levels).values / 100  # Convert % to fraction
            pressure = np.array(levels)[:, np.newaxis, np.newaxis]  # Expand dimensions to match other arrays
            q_provided = ds_upper['q'].sel(isobaricInhPa=levels).values
            
            e_s = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))
            q_calculated = (relative_humidity * e_s) / (pressure - (1 - relative_humidity) * e_s)
            # 根据NOAA公式计算饱和水汽压 e_s
            e_s = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))

            # 根据NOAA公式计算比湿 q
            q_calculated = 0.622 * (relative_humidity * e_s) / (pressure - (relative_humidity * e_s))
            difference = np.abs(q_provided - q_calculated)
            max_difference = np.max(difference)
            mean_difference = np.mean(difference)
            
            print(f"Max difference between provided q and calculated q: {max_difference}")
            print(f"Mean difference between provided q and calculated q: {mean_difference}")
        else:
            print(f"File {grib_file_path} does not exist")

    compare_q_values(output_dir, grib_filename)

if __name__ == '__main__':
    output_dir = 'input'
    datetime = '2021041700'
    get_gfs_data(output_dir, datetime)