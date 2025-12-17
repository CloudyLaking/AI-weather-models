# AI-weather-models\Pangu\Main\get-data-pangu.py
# Download and convert meteorological data (ERA5, GFS, ECMWF) to NPY format

import warnings
import sys
import io
from contextlib import redirect_stderr

# 抑制 cfgrib 和 xarray 的警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*DatasetBuildError.*')

import cdsapi
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import xarray as xr
import os
import numpy as np
from datetime import datetime, timedelta

class DataDownloader:
    """Unified data downloader and converter supporting ERA5, GFS, and EC data"""
    
    @staticmethod
    def print_input_stats(input_dir):
        """
        打印已生成的 input_surface.npy 和 input_upper.npy 文件中各变量的统计信息
        
        Args:
            input_dir: 输入数据目录
        """
        surface_file = os.path.join(input_dir, 'input_surface.npy')
        upper_file = os.path.join(input_dir, 'input_upper.npy')
        
        # 打印 surface 数据统计
        if os.path.exists(surface_file):
            surface_data = np.load(surface_file)
            var_names = ['mslet', 'u10', 'v10', 't2m']
            print(f"[INPUT] Surface variables (shape: {surface_data.shape}):")
            for i, var_name in enumerate(var_names):
                print(f"  ✓ {var_name}: min={surface_data[i].min():.2f}, max={surface_data[i].max():.2f}")
        
        # 打印 upper 数据统计
        if os.path.exists(upper_file):
            upper_data = np.load(upper_file)
            var_names = ['gh', 'q', 't', 'u', 'v']
            print(f"[INPUT] Upper variables (shape: {upper_data.shape}):")
            for i, var_name in enumerate(var_names):
                print(f"  ✓ {var_name}: min={upper_data[i].min():.2f}, max={upper_data[i].max():.2f}")
    
    def __init__(self, output_dir='Input/pangu', input_dir='Input/pangu', raw_input_dir='Input/Pangu_raw'):
        """
        Initialize data downloader
        
        Args:
            output_dir: Directory for processed input data (for model use) - DEPRECATED, use input_dir instead
            input_dir: Directory for processed input data (NPY files)
            raw_input_dir: Directory for raw downloaded data (NC/GRIB files)
        """
        self.output_dir = input_dir  # Keep for compatibility, but use processed input directory
        self.input_dir = input_dir   # Processed input directory
        self.raw_input_dir = raw_input_dir  # Raw download directory
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.raw_input_dir, exist_ok=True)
    
    @staticmethod
    def download_with_progress(url, target):
        """
        Download file with progress bar
        
        Args:
            url: Download URL
            target: Target file path
            
        Returns:
            bool: Whether download was successful
        """
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        try:
            response = session.get(url, stream=True, verify=False)
            if response.status_code != 200:
                print(f"[ERROR] Download failed, status code: {response.status_code}")
                return False
                
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            t = tqdm(total=total_size, unit='iB', unit_scale=True)
            with open(target, 'wb') as file:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    file.write(data)
            t.close()
            
            if total_size != 0 and t.n != total_size:
                print("[ERROR] Download error")
                return False
            print(f"[OK] Download completed: {target}")
            return True
        except Exception as e:
            print(f"[ERROR] Download exception: {e}")
            return False
    
    def convert_nc_to_npy(self, nc_filename, npy_filename, reshape=False):
        """
        Convert NetCDF file to NPY array
        
        Args:
            nc_filename: Source NetCDF filename
            npy_filename: Target NPY filename
            reshape: Whether to reshape data
        """
        nc_file_path = os.path.join(self.raw_input_dir, nc_filename)
        if not os.path.exists(nc_file_path):
            print(f"[ERROR] File not found: {nc_file_path}")
            return False
            
        try:
            ds = xr.open_dataset(nc_file_path)
            data = ds.to_array().values
            
            if reshape:
                if data.shape[1] == 1 and len(data.shape) == 5:
                    data = data.reshape(data.shape[0], data.shape[2], data.shape[3], data.shape[4])
                else:
                    data = data.reshape(data.shape[0], data.shape[2], data.shape[3])
            
            npy_file_path = os.path.join(self.input_dir, npy_filename)
            np.save(npy_file_path, data)
            print(f"[OK] Converted: {npy_file_path}, shape: {data.shape}")
            return True
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return False
    
    def convert_grib_to_npy(self, grib_filename, surface_npy_filename, upper_npy_filename):
        """
        Convert GRIB file to NPY array (for GFS data)
        Based on the correct implementation from AImodels-raw
        
        Args:
            grib_filename: Source GRIB filename
            surface_npy_filename: Target surface layer NPY filename
            upper_npy_filename: Target upper level NPY filename
        """
        grib_file_path = os.path.join(self.raw_input_dir, grib_filename)
        if not os.path.exists(grib_file_path):
            print(f"[ERROR] File not found: {grib_file_path}")
            return False
            
        try:
            # Pangu模型要求的标准13个气压层
            levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
            
            # 抑制cfgrib的stderr输出（避免打印DatasetBuildError和skipping variable消息）
            with redirect_stderr(io.StringIO()):
                ds_mean_sea = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'})
                ds_upper = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'}, errors='ignore')
                ds_sigma = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'sigma'})
            
            # ===== 处理表面数据 =====
            # 表面变量：mslet(海平面气压)、u(10m U风)、v(10m V风)、t(2m温度)
            # mslet来自meanSea层，u/v/t来自sigma层
            surface_data = []
            
            # 获取海平面气压
            if 'mslet' in ds_mean_sea:
                mslet_data = ds_mean_sea['mslet'].values
                surface_data.append(mslet_data)
            else:
                print("[ERROR] mslet variable not found")
                return False
            
            # 获取sigma层的u, v, t (10m风 和 2m温度来自sigma层)
            surface_sigma_vars = ['u', 'v', 't']
            for var in surface_sigma_vars:
                if var in ds_sigma:
                    var_data = ds_sigma[var].values
                    surface_data.append(var_data)
                else:
                    print(f"[ERROR] {var} variable not found in sigma layer")
                    return False
            
            surface_array = np.stack(surface_data, axis=0)
            surface_path = os.path.join(self.input_dir, surface_npy_filename)
            np.save(surface_path, surface_array)
            print(f"[OK] Surface layer data saved: {surface_path}, shape: {surface_array.shape}")            
            # ===== 处理上层大气数据 =====
            # 上层变量：gh(地位势)、r(相对湿度)、t(温度)、u(U风)、v(V风)
            upper_vars = ['gh', 'r', 't', 'u', 'v']
            upper_data = []
            
            for var in upper_vars:
                if var in ds_upper:
                    # 直接选择目标气压层
                    data = ds_upper[var].sel(isobaricInhPa=levels).values
                    
                    # 地位势需要转换为地位势能（乘以重力加速度）
                    if var == 'gh':
                        data = data * 9.80665
                    
                    # 检查高度层顺序：如果第一个值大于最后一个值，说明顺序反了（高空到地面）
                    # 需要反转到地面到高空的顺序
                    if data[0, 0, 0] > data[-1, 0, 0]:
                        print(f"  [INFO] Reversing {var} pressure levels (high to low -> low to high)")
                        data = data[::-1]
                    
                    upper_data.append(data)
                else:
                    print(f"[ERROR] Upper level {var} variable not found")
                    return False
            
            # 从相对湿度计算比湿（使用NOAA公式）
            # q = 0.622 * (RH * e_s) / (P - RH * e_s)
            # 其中 e_s = 6.112 * exp((17.67 * (T - 273.15)) / (T - 29.65))
            temperature = ds_upper['t'].sel(isobaricInhPa=levels).values  # shape: (13, lat, lon)
            relative_humidity = ds_upper['r'].sel(isobaricInhPa=levels).values / 100  # 转换百分比为小数
            
            # 扩展pressure维度以匹配其他数组 (13, 1, 1) -> 可广播
            pressure = np.array(levels, dtype=np.float32)[:, np.newaxis, np.newaxis]
            
            # 计算饱和水汽压
            e_s = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))
            
            # 计算比湿
            q = 0.622 * (relative_humidity * e_s) / (pressure - (relative_humidity * e_s))
            
            # 用计算的比湿替换相对湿度
            upper_data[upper_vars.index('r')] = q
            
            # 堆叠数据
            upper_array = np.stack(upper_data, axis=0)  # shape: (5, 13, lat, lon)
            
            upper_path = os.path.join(self.input_dir, upper_npy_filename)
            np.save(upper_path, upper_array)
            print(f"[OK] Upper level data saved: {upper_path}, shape: {upper_array.shape}")
            return True
            
        except Exception as e:
            print(f"[ERROR] GRIB conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_era5_data(self, datetime_str):
        """
        Acquire and process ERA5 reanalysis data
        
        Args:
            datetime_str: Date time string in format 'YYYYMMDDHH'
        """
        print(f"\n[INFO] Starting ERA5 data acquisition ({datetime_str})...")
        
        try:
            client = cdsapi.Client()
        except Exception as e:
            print(f"[ERROR] CDS client initialization failed: {e}")
            return False
        
        year = datetime_str[:4]
        month = datetime_str[4:6]
        day = datetime_str[6:8]
        time = datetime_str[8:10]
        
        surface_output_file = os.path.join(self.raw_input_dir, f'era5_surface_{datetime_str}.nc')
        if not os.path.exists(surface_output_file):
            print(f"[INFO] Downloading ERA5 surface data...")
            surface_dataset = "reanalysis-era5-single-levels"
            surface_request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "mean_sea_level_pressure",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                    "2m_temperature",
                ],
                "year": year,
                "month": month,
                "day": day,
                "time": time,
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            try:
                result = client.retrieve(surface_dataset, surface_request)
                result.download(surface_output_file)
                print(f"[OK] ERA5 surface data downloaded")
            except Exception as e:
                print(f"[ERROR] ERA5 surface data download failed: {e}")
                return False
        else:
            print(f"[OK] ERA5 surface data file already exists")
        
        upper_output_file = os.path.join(self.raw_input_dir, f'era5_upper_{datetime_str}.nc')
        if not os.path.exists(upper_output_file):
            print(f"[INFO] Downloading ERA5 upper level data...")
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
                "year": year,
                "month": month,
                "day": day,
                "time": time,
                "pressure_level": [
                    "50", "100", "150", "200", "250", "300",
                    "400", "500", "600", "700", "850", "925", "1000"
                ],
                "data_format": "netcdf",
                "download_format": "unarchived"
            }
            
            try:
                result = client.retrieve(upper_dataset, upper_request)
                result.download(upper_output_file)
                print(f"[OK] ERA5 upper level data downloaded")
            except Exception as e:
                print(f"[ERROR] ERA5 upper level data download failed: {e}")
                return False
        else:
            print(f"[OK] ERA5 upper level data file already exists")
        
        self.convert_nc_to_npy(f'era5_surface_{datetime_str}.nc', 'input_surface.npy', reshape=True)
        self.convert_nc_to_npy(f'era5_upper_{datetime_str}.nc', 'input_upper.npy', reshape=True)
        self.print_input_stats(self.input_dir)
        return True
    
    def get_gfs_data(self, datetime_str):
        """
        Acquire and process GFS forecast data
        
        Args:
            datetime_str: Date time string in format 'YYYYMMDDHH'
        """
        print(f"\n[INFO] Starting GFS data acquisition ({datetime_str})...")
        
        current_date = datetime.now()
        input_date = datetime.strptime(datetime_str, '%Y%m%d%H')
        days_diff = (current_date - input_date).days
        
        if 0 <= days_diff <= 5:
            base_url = 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod'
            date = datetime_str[:8]
            hour = datetime_str[8:10]
            filename = f'gfs.t{hour}z.pgrb2.0p25.f000'
            url = f'{base_url}/gfs.{date}/{hour}/atmos/{filename}'
        else:
            base_url = 'https://thredds.rda.ucar.edu/thredds/fileServer/files/g/d084001'
            year = datetime_str[:4]
            date = datetime_str[:8]
            filename = f'gfs.0p25.{datetime_str}.f000.grib2'
            url = f'{base_url}/{year}/{date}/{filename}'
        
        grib_filename = f'gfs_{datetime_str}.grib2'
        grib_file_path = os.path.join(self.raw_input_dir, grib_filename)
        
        if not os.path.exists(grib_file_path):
            print(f"[INFO] Downloading GFS data: {url}")
            if not self.download_with_progress(url, grib_file_path):
                return False
        else:
            print(f"[OK] GFS file already exists: {grib_file_path}")
        
        if self.convert_grib_to_npy(grib_filename, 'input_surface.npy', 'input_upper.npy'):
            self.print_input_stats(self.input_dir)
            return True
        return False
    
    def get_ec_data(self, datetime_str):
        """
        Acquire and process ECMWF Open Data (forecast data)
        
        Args:
            datetime_str: Date time string in format 'YYYYMMDDHH'
            
        Note: Uses ECMWF Open Data API (no key required for recent data)
        """
        print(f"\n[INFO] Starting ECMWF Open Data acquisition ({datetime_str})...")
        
        try:
            import earthkit.data as ekd
        except ImportError:
            print("[ERROR] earthkit package not installed, run: pip install earthkit-data earthkit-regrid")
            return False
        
        # Set earthkit cache directory
        cache_dir = os.path.join(self.raw_input_dir, '.earthkit_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir_posix = cache_dir.replace('\\', '/')
        os.environ['EARTHKIT_CACHE_DIR'] = cache_dir_posix
        
        try:
            date_obj = datetime.strptime(datetime_str, '%Y%m%d%H')
            
            # Download surface parameters from ECMWF Open Data
            # Parameters: 2m temperature (2t), 10m wind (10u, 10v), mean sea level pressure (msl)
            print(f"[STEP 1/2] Downloading ECMWF Open Data surface parameters...")
            
            data = ekd.from_source(
                "ecmwf-open-data",
                date=date_obj,
                param=["2t", "10u", "10v", "msl"]
                # Surface level only (no levelist needed)
            )
            
            # Convert to numpy array
            print(f"[STEP 2/2] Converting to numpy format...")
            # Data is an iterable, extract each parameter
            
            fields = {}
            for field in data:
                param_name = field.metadata('param')
                values = field.to_numpy()
                fields[param_name] = values
                print(f"  [OK] Loaded {param_name}: shape {values.shape}")
            
            # Extract surface data in the correct order for Pangu
            # Pangu expects [4, H, W] where 4 channels are [mslp, u10, v10, t2m]
            t2m = fields.get("2t")
            u10 = fields.get("10u")
            v10 = fields.get("10v")
            mslp = fields.get("msl") / 100  # Convert Pa to hPa
            
            if any(v is None for v in [t2m, u10, v10, mslp]):
                missing = [k for k, v in {"2t": t2m, "10u": u10, "10v": v10, "msl": mslp}.items() if v is None]
                raise ValueError(f"Missing parameters: {missing}")
            
            # Create output surface array [4, H, W]
            input_surface = np.stack([mslp, u10, v10, t2m], axis=0)
            
            # For upper atmosphere, we need to provide dummy data or use climatology
            # Pangu model expects [5, 13, H, W] for upper atmosphere
            # (5 parameters: gh, q, t, u, v; 13 pressure levels)
            print(f"[INFO] Creating upper atmosphere placeholder...")
            
            # Placeholder for upper atmosphere with correct dimensions [5, 13, H, W]
            # 5 parameters: gh, q, t, u, v
            # 13 pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
            input_upper = np.zeros((5, 13, input_surface.shape[1], input_surface.shape[2]))
            
            # Save to NPY files
            surface_file = os.path.join(self.input_dir, 'input_surface.npy')
            upper_file = os.path.join(self.input_dir, 'input_upper.npy')
            
            np.save(surface_file, input_surface)
            np.save(upper_file, input_upper)
            
            print(f"[OK] ECMWF Open Data processed and saved")
            self.print_input_stats(self.input_dir)
            return True
            
        except Exception as e:
            print(f"[ERROR] ECMWF data download/processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def get_data(data_source, datetime_str):
    """
    Convenience function: One-click data acquisition
    
    Args:
        data_source: Data source, 'ERA5', 'GFS' or 'EC'
        datetime_str: Date time string in format 'YYYYMMDDHH'
        
    Returns:
        bool: Whether data acquisition and processing was successful
    """
    downloader = DataDownloader()
    
    if data_source.upper() == 'ERA5':
        return downloader.get_era5_data(datetime_str)
    elif data_source.upper() == 'GFS':
        return downloader.get_gfs_data(datetime_str)
    elif data_source.upper() == 'EC':
        return downloader.get_ec_data(datetime_str)
    else:
        print(f"[ERROR] Unsupported data source: {data_source}")
        return False


if __name__ == '__main__':
    # Example usage
    get_data('GFS', '2025121200')
