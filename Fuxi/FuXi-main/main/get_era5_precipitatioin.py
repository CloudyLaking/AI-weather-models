# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 15:15:01 2023

@author: rtdhthf
"""

import cdsapi
from datetime import datetime, timedelta
import numpy as np

c = cdsapi.Client()


# days_num = np.arange(1, 32, 1)
# days = []
# for ii in range(0, days_num.size):
#     days.append('%02d'%days_num[ii])




c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'total_precipitation',
        'year': ['2020'], #['2021', '2022', '2023'],
        'month': ['10'], #['06', '07', '08', '09', '10'],
        'day': ['04', '05', '06', '07', '08', '09', '10'],#days,
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
    },
    'ERA5_3yr_JJASO_precipitation.nc')



        
