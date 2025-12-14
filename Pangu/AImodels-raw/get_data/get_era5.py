import cdsapi
import os
import xarray as xr
import numpy as np
from tqdm import tqdm
import requests

# 下载文件并显示进度条
def download_with_progress_bar(url, target):
    response = requests.get(url, stream=True, verify=False)  # 禁用 SSL 验证
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(target, 'wb') as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR, something went wrong")

# 重写 retrieve 方法以添加进度条
def retrieve_with_progress_bar(client, name, request, target=None):
    original_retrieve = client.retrieve
    result = original_retrieve(name, request)
    download_url = result.location
    download_with_progress_bar(download_url, target)

# 读取指定的.nc文件并转换为.npy数组
def convert_nc_to_npy(output_dir, nc_filename, npy_filename, reshape=False):
    nc_file_path = os.path.join(output_dir, nc_filename)
    if os.path.exists(nc_file_path):
        ds = xr.open_dataset(nc_file_path)
        data = ds.to_array().values
        if reshape:
            if data.shape[1] == 1 and len(data.shape) == 5:  # 检查是否为高空数据
                data = data.reshape(data.shape[0], data.shape[2], data.shape[3], data.shape[4])
            else:
                data = data.reshape(data.shape[0], data.shape[2], data.shape[3])
        npy_file_path = os.path.join('input', npy_filename)
        np.save(npy_file_path, data)
        print(f"Converted {nc_file_path} to {npy_file_path}")
    else:
        print(f"File {nc_file_path} does not exist")

# 下载并处理地面数据
def process_surface_data(client, output_dir, year, month, day, time):
    surface_output_file = os.path.join(output_dir, 'era5_surface_data.nc')
    surface_dataset = "reanalysis-era5-single-levels"
    surface_request = {
        "product_type": ["reanalysis"],
        "variable": [
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    retrieve_with_progress_bar(client, surface_dataset, surface_request, target=surface_output_file)
    convert_nc_to_npy(output_dir, 'era5_surface_data.nc', 'input_surface.npy', reshape=True)

# 下载并处理高空数据
def process_upper_data(client, output_dir, year, month, day, time):
    upper_output_file = os.path.join(output_dir, 'era5_upper_data.nc')
    upper_dataset = "reanalysis-era5-pressure-levels"
    upper_request = {
        "product_type": ["reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [year],
        "month": [month],
        "day": [day],
        "time": [f"{time}:00"],
        "pressure_level": [
            "50", "100", "150",
            "200", "250", "300",
            "400", "500", "600",
            "700", "850", "925",
            "1000"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    retrieve_with_progress_bar(client, upper_dataset, upper_request, target=upper_output_file)
    convert_nc_to_npy(output_dir, 'era5_upper_data.nc', 'input_upper.npy', reshape=True)

def main():
    output_dir = 'raw_data'
    client = cdsapi.Client()

    inputs = '2013110712'
    year = inputs[:4]
    month = inputs[4:6]
    day = inputs[6:8]
    time = inputs[8:]
    
    # 处理地面数据
    process_surface_data(client, output_dir, year, month, day, time)

    # 处理高空数据
    process_upper_data(client, output_dir, year, month, day, time)

if __name__ == '__main__':
    main()