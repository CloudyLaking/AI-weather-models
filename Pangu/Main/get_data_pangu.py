# AI-weather-models\Pangu\Main\get-data-pangu.py
# Download and convert meteorological data (ERA5, GFS, ECMWF) to NPY format

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
    
    def __init__(self, output_dir='../../../Output/pangu', input_dir='../../../Input/pangu', raw_input_dir='../../../Input/Pangu_raw'):
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
            print(f"[OK] Converted: {nc_file_path} -> {npy_file_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return False
    
    def convert_grib_to_npy(self, grib_filename, surface_npy_filename, upper_npy_filename):
        """
        Convert GRIB file to NPY array (for GFS data)
        
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
            ds_mean_sea = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'meanSea'})
            ds_upper = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
            ds_sigma = xr.open_dataset(grib_file_path, engine='cfgrib', filter_by_keys={'typeOfLevel': 'sigma'})
            
            surface_vars = ['mslet', 'u', 'v', 't']
            surface_data = []
            
            if 'mslet' in ds_mean_sea:
                surface_data.append(ds_mean_sea['mslet'].values)
            else:
                print("[ERROR] mslet variable not found")
                return False
            
            for var in surface_vars[1:]:
                if var in ds_sigma:
                    surface_data.append(ds_sigma[var].values)
                else:
                    print(f"[ERROR] {var} variable not found")
                    return False
            
            surface_array = np.stack(surface_data, axis=0)
            surface_path = os.path.join(self.input_dir, surface_npy_filename)
            np.save(surface_path, surface_array)
            print(f"[OK] Surface layer data saved: {surface_path}")
            
            upper_vars = ['gh', 'q', 't', 'u', 'v']
            upper_data = []
            
            for var in upper_vars:
                if var in ds_upper:
                    data = ds_upper[var].values
                    upper_data.append(data)
                else:
                    print(f"[ERROR] Upper level {var} variable not found")
                    return False
            
            upper_array = np.stack(upper_data, axis=0)
            upper_path = os.path.join(self.input_dir, upper_npy_filename)
            np.save(upper_path, upper_array)
            print(f"[OK] Upper level data saved: {upper_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] GRIB conversion failed: {e}")
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
        
        return self.convert_grib_to_npy(grib_filename, 'input_surface.npy', 'input_upper.npy')
    
    def get_ec_data(self, datetime_str):
        """
        Acquire and process ECMWF/AIFS forecast data
        
        Args:
            datetime_str: Date time string in format 'YYYYMMDDHH'
            
        Note: Requires ECMWF API key configuration
        """
        print(f"\n[INFO] Starting ECMWF data acquisition ({datetime_str})...")
        print("[WARN] ECMWF data requires API key configuration, see official documentation")
        
        try:
            from ecmwfapi import ECMWFDataServer
        except ImportError:
            print("[ERROR] ecmwf-api-client not installed, run: pip install ecmwf-api-client")
            return False
        
        year = datetime_str[:4]
        month = datetime_str[4:6]
        day = datetime_str[6:8]
        time = datetime_str[8:10]
        
        output_file = os.path.join(self.raw_input_dir, f'ecmwf_{datetime_str}.grib')
        
        if os.path.exists(output_file):
            print(f"[OK] ECMWF file already exists: {output_file}")
        else:
            try:
                server = ECMWFDataServer()
                server.retrieve({
                    "class": "od",
                    "dataset": "reanalysis-complete",
                    "date": f"{year}-{month}-{day}",
                    "time": f"{time}:00:00",
                    "levtype": "sfc",
                    "param": "151/165/166/167",
                    "step": "0",
                    "stream": "oper",
                    "type": "an",
                    "target": output_file
                })
                print(f"[OK] ECMWF data downloaded")
            except Exception as e:
                print(f"[ERROR] ECMWF data download failed: {e}")
                return False
        
        return self.convert_grib_to_npy(f'ecmwf_{datetime_str}.grib', 'input_surface.npy', 'input_upper.npy')


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
