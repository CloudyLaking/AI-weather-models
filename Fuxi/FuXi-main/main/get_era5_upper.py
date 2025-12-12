# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:15:01 2023

@author: rtdhthf
"""

import cdsapi
from datetime import datetime, timedelta
# import pandas as pd




c = cdsapi.Client(timeout=600)

yr = ['2020']

month = ['10']

day = ['05', '06', '07', '08', '09', '10'] 

time =  ['00:00', '06:00', '12:00', '18:00']

# all_need_times = []
# d = datetime(2021+yr_idx,7,1,0,0,0)
# while d <= datetime(2021+yr_idx,10,31,18,0,0):
#     all_need_times.append(d)
#     d = d + timedelta(hours = 6)


c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            'geopotential', 'specific_humidity', 'temperature',
            'u_component_of_wind', 'v_component_of_wind', 'relative_humidity'
        ],
        'pressure_level': [
                '1','2','3',
                '5','7','10',
                '20','30','50',
                '70','100','125',
                '150','175','200',
                '225','250','300',
                '350','400','450',
                '500','550','600',
                '650','700','750',
                '775','800','825',
                '850','875','900',
                '925','950','975',
                '1000'
        ],
        # 'area': [90, 0, 0, 180,],
        'year': yr,
        'month': month,
        'day': day,
        'time': time,
        'format': 'grib',
    },
    'ERA5_pl_case_026.grib')
    
    

# nohup python -u get_era5_upper.py > get_era5_upper.out 2>&1 &
        
        
        
        
        
