import xarray as xr
import cfgrib
import sys

def print_grib_info(filename):
    type_of_levels = [
        'planetaryBoundaryLayer', 'surface', 'isobaricInhPa', 'meanSea',
        'depthBelowLandLayer', 'heightAboveGround', 'atmosphereSingleLayer',
        'heightAboveGroundLayer', 'tropopause', 'maxWind', 'heightAboveSea',
        'isothermZero', 'highestTroposphericFreezing', 'pressureFromGroundLayer',
        'sigmaLayer', 'sigma', 'potentialVorticity'
    ]
    
    for level in type_of_levels:
        try:
            ds = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': level})
            print(f"Variables for typeOfLevel={level}:")
            print(ds)
        except Exception as e:
            print(f"Could not open dataset for typeOfLevel={level}: {e}")

def print_specific_level_variables(filename, level):
    try:
        ds = xr.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': level})
        print(f"{level.capitalize()} level variables:")
        print(list(ds.variables))
    except Exception as e:
        print(f"Could not open dataset for typeOfLevel={level}: {e}")

if __name__ == '__main__':
    filename = r'Input\Pangu_raw\gfs_2025121400.grib2'
    print_grib_info(filename)
    print_specific_level_variables(filename, 'heightAboveGround')
    print_specific_level_variables(filename, 'surface')