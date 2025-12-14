import os
import xarray as xr
from tqdm import tqdm
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_gdas_data(output_dir, datetime):

    # 下载的url
    #           https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d083003/2021/202104/gdas1.fnl0p25.2021041700.f00.grib2
    base_url = 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d083003'
    year = datetime[:4]
    mon = datetime[:6]
    url = f'{base_url}/{year}/{mon}/gdas1.fnl0p25.{datetime}.f00.grib2'

    grib_filename = f'gdas_{datetime}.grib2'
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

    # 读取grib文件并转换为npy数组
    def convert_grib_to_npy_and_process(output_dir, grib_filename, surface_npy_filename, upper_npy_filename):
        grib_file_path = os.path.join(output_dir, grib_filename)
        if os.path.exists(grib_file_path):
            ds_mean_sea = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'})
            ds_upper = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, errors='ignore')
            ds_sigma = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'sigma'})
            
            surface_vars = [
                'mslet',  # Mean sea level pressure  prmsl mslet
                'u',    # U component of wind at 10 meters
                'v',    # V component of wind at 10 meters
                't'     # Temperature at 2 meters
            ]
            surface_data = []
            if 'mslet' in ds_mean_sea:
                data = ds_mean_sea['mslet'].values
                surface_data.append(data)
            else:
                print(f"Variable {var} not found in {grib_file_path}")
                return
            for var in surface_vars[1:]:
                if var in ds_sigma:
                    data = ds_sigma[var].values
                    surface_data.append(data)
                else:
                    print(f"Variable {var} not found in {grib_file_path}")
                    return
            
            surface_data = np.stack(surface_data)
            surface_data = surface_data.reshape(surface_data.shape[0], surface_data.shape[1], surface_data.shape[2])
            np.save(os.path.join('input', surface_npy_filename), surface_data)
            print(f"Converted {grib_file_path} to input/{surface_npy_filename}")
            
            upper_vars = [
                'gh',  # Geopotential height
                'r',   # Relative humidity
                't',   # Temperature
                'u',   # U component of wind
                'v'    # V component of wind
            ]

            upper_data = []
            levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            
            for var in upper_vars:
                if var in ds_upper:
                    data = ds_upper[var].sel(isobaricInhPa=levels).values
                    if var == 'gh':
                        data = data * 9.80665
                    upper_data.append(data)
                else:
                    print(f"Variable {var} not found in {grib_file_path}")
                    return
            
            temperature = ds_upper['t'].sel(isobaricInhPa=levels).values
            relative_humidity = ds_upper['r'].sel(isobaricInhPa=levels).values / 100  # Convert % to fraction
            pressure = np.array(levels)[:, np.newaxis, np.newaxis]  # Expand dimensions to match other arrays
            
            # 根据NOAA公式计算比湿 q
            e_s = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))
            q = 0.622 * (relative_humidity * e_s) / (pressure - (relative_humidity * e_s))
            upper_data[upper_vars.index('r')] = q
            
            upper_data = np.stack(upper_data)
            upper_data = upper_data.reshape(upper_data.shape[0], upper_data.shape[1], upper_data.shape[2], upper_data.shape[3])
            np.save(os.path.join('input', upper_npy_filename), upper_data)
            print(f"Converted {grib_file_path} to input/{upper_npy_filename}")
        else:
            print(f"File {grib_file_path} does not exist")

    convert_grib_to_npy_and_process(output_dir, grib_filename, 'input_surface.npy', 'input_upper.npy')

if __name__ == '__main__':
    output_dir = 'input'
    datetime = '2021041700'
    get_gdas_data(output_dir, datetime)