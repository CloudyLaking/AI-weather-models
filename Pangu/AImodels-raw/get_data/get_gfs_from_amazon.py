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

# 读取指定的.nc文件并打印变量信息
def print_nc_variables(nc_file_path):
    if os.path.exists(nc_file_path):
        ds = xr.open_dataset(nc_file_path)
        print(f"Variables in {nc_file_path}:")
        print(ds.variables)
    else:
        print(f"File {nc_file_path} does not exist")

def main():
    url = 'https://noaa-gfs-warmstart-pds.s3.amazonaws.com/gfs.20210417/00/RESTART/20210416.210000.sfcanl_data.tile1.nc'
    output_dir = 'raw_data'
    nc_filename = 'sfcanl_data.tile1.nc'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    nc_file_path = os.path.join(output_dir, nc_filename)
    
    # 下载.nc文件
    download_with_progress_bar(url, nc_file_path)
    
    # 打印.nc文件中的变量信息
    print_nc_variables(nc_file_path)

if __name__ == '__main__':
    main()