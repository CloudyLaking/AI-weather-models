# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:15:01 2023

@author: rtdhthf
"""

import cdsapi
from datetime import datetime, timedelta


c = cdsapi.Client(timeout=600)

yr = ['2020']

month = ['10']

day = ['05', '06', '07', '08', '09', '10'] 

time =  ['00:00', '06:00', '12:00', '18:00']



c.retrieve(
'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [#'total_precipitation',
             '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
             'mean_sea_level_pressure', 'sea_surface_temperature', 'skin_temperature',
             'snow_albedo', 'snow_depth', 'snowfall',
             'soil_temperature_level_1', 'soil_type', 'surface_pressure',
             'total_precipitation','soil_temperature_level_2',
             'soil_temperature_level_3','soil_temperature_level_4',
            'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
            'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
        ],
        'year': yr,
        'month': month,
        'day': day,
        'time': time,
        'format': 'grib',
    },
    'ERA5_srf_case_026_test.grib')
    
    
# nohup python -u get_era5_surface.py > get_era5_surface.out 2>&1 &
        
        
        
        
