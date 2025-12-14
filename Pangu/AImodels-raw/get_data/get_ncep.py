import os
import xarray as xr
from tqdm import tqdm
import requests
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_ncep_data(output_dir, datetime):
    base_url = 'https://thredds.rda.ucar.edu/thredds/ncss/grid/aggregations/g/d083003/1/TP'
    params = (
        '?var=Geopotential_height_isobaric'
        '&var=Relative_humidity_isobaric'
        '&var=Temperature_isobaric'
        '&var=u-component_of_wind_isobaric'
        '&var=v-component_of_wind_isobaric'

        '&var=MSLP_Eta_model_reduction_msl'
        '&var=u-component_of_wind_planetary_boundary'
        '&var=v-component_of_wind_planetary_boundary'
        '&var=Temperature_surface'
        
        '&north=90.000'
        '&west=-.125'
        '&east=-.125'
        '&south=-90.000'
        '&horizStride=1'
        f'&time={datetime[:4]}-{datetime[4:6]}-{datetime[6:8]}T{datetime[8:]}:00:00Z'
        '&vertCoord=1'
        '&accept=netcdf3'
    )
    url = base_url + params

    nc_filename = f'ncep_data_{datetime}.nc'
    nc_file_path = os.path.join(output_dir, nc_filename)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
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

    if not os.path.exists(nc_file_path):
        download_with_progress_bar(url, nc_file_path)
    else:
        print(f"File {nc_file_path} already exists, skipping download.")

    def convert_nc_to_npy_and_process(output_dir, nc_filename, surface_npy_filename, upper_npy_filename):
        nc_file_path = os.path.join(output_dir, nc_filename)
        if os.path.exists(nc_file_path):
            ds = xr.open_dataset(nc_file_path, engine='netcdf4')
            print(f"Dataset variables: {list(ds.variables.keys())}")
            
            surface_vars = [
                'Temperature_surface',
                'MSLP_Eta_model_reduction_msl',
                'u-component_of_wind_planetary_boundary',
                'v-component_of_wind_planetary_boundary'
            ]
            surface_data = []
            for var in surface_vars:
                if var in ds:
                    data = ds[var].values
                    print(f"{var} data: {data}")
                    surface_data.append(data)
                else:
                    print(f"Variable {var} not found in {nc_file_path}")
                    return
            surface_data = np.stack(surface_data)
            surface_data = surface_data.reshape(surface_data.shape[0], surface_data.shape[2], surface_data.shape[3])
            np.save(os.path.join('input', surface_npy_filename), surface_data)
            print(f"Converted {nc_file_path} to input/{surface_npy_filename}")
            
            upper_vars = [
                'Geopotential_height_isobaric',
                'Relative_humidity_isobaric',
                'Temperature_isobaric',
                'u-component_of_wind_isobaric',
                'v-component_of_wind_isobaric'
            ]
            upper_data = []
            for var in upper_vars:
                if var in ds:
                    data = ds[var].values
                    print(f"{var} data: {data}")
                    if var == 'Geopotential_height_isobaric':
                        data = data * 9.80665
                    upper_data.append(data)
                else:
                    print(f"Variable {var} not found in {nc_file_path}")
                    return
            
            temperature = ds['Temperature_isobaric'].values
            relative_humidity = ds['Relative_humidity_isobaric'].values
            pressure = ds['Geopotential_height_isobaric'].values / 9.80665

            e_s = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))
            q = (relative_humidity * e_s) / (pressure - (1 - relative_humidity) * e_s)
            
            upper_data[upper_vars.index('Relative_humidity_isobaric')] = q
            
            upper_data = np.stack(upper_data)
            upper_data = upper_data.reshape(upper_data.shape[0], upper_data.shape[2], upper_data.shape[3], upper_data.shape[4])
            np.save(os.path.join('input', upper_npy_filename), upper_data)
            print(f"Converted {nc_file_path} to input/{upper_npy_filename}")
        else:
            print(f"File {nc_file_path} does not exist")

    convert_nc_to_npy_and_process(output_dir, nc_filename, 'input_surface.npy', 'input_upper.npy')

if __name__ == '__main__':
    output_dir = 'input'
    datetime = '2023031309'
    get_ncep_data(output_dir, datetime)