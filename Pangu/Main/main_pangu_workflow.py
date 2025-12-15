# AI-weather-models\Pangu\Main\main-pangu-workflow.py
# Integrated workflow orchestrating data acquisition, model execution, and visualization

import os
import sys
import argparse
from datetime import datetime, timedelta
import numpy as np
import importlib.util
import warnings

# 抑制临时文件清理错误（Windows特定）
import atexit
def _suppress_tempfile_cleanup_errors():
    """抑制 earthkit 退出时的临时文件清理权限错误"""
    original_atexit_register = atexit.register
    def patched_atexit_register(func, *args, **kwargs):
        def wrapped(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except (PermissionError, OSError, NotADirectoryError):
                pass  # 忽略文件清理错误
        return original_atexit_register(wrapped, *args, **kwargs)
    atexit.register = patched_atexit_register

_suppress_tempfile_cleanup_errors()

# Import from local modules
try:
    from get_data_pangu import DataDownloader, get_data
    from run_pangu import PanguRunner, run_pangu_forecast
    from draw_pangu_results import PanguResultDrawer, draw_pangu_results
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    print("[INFO] Make sure get-data-pangu.py, run-pangu.py, and draw-pangu-results.py are in the same directory")
    sys.exit(1)


def run_complete_workflow(data_source='GFS', init_datetime_str=None, model_type=24, 
                         run_times=8, lon_range=None, lat_range=None,
                         draw_results=True, skip_existing=True,
                         input_dir='../../../Input/Pangu',
                         raw_input_dir='../../../Input/Pangu_raw',
                         output_dir='../../../Output/Pangu',
                         model_dir='../../../Models-weights/Pangu',
                         image_dir='../../../Run-output-png/Pangu'):
    """
    Execute complete Pangu weather forecast workflow
    
    Args:
        data_source: Data source - 'GFS', 'ERA5', or 'ECMWF'
        init_datetime_str: Initialization datetime in format 'YYYYMMDDHH'
                          If None, use current time
        model_type: Model type - 6 for 6h, 24 for 24h forecast
        run_times: Number of forecast rounds
        lon_range: Geographic longitude range [lon_min, lon_max]
        lat_range: Geographic latitude range [lat_min, lat_max]
        draw_results: Whether to draw visualization results
        skip_existing: Whether to skip existing data and outputs
        input_dir: Processed input data directory (NPY files)
        raw_input_dir: Raw input data directory (NC/GRIB files)
        output_dir: Output data directory (model predictions)
        model_dir: Model weights directory
        image_dir: Visualization output directory
        
    Returns:
        dict: Workflow execution results
    """
    
    # Get the script's directory and convert relative paths based on it
    script_dir = os.path.dirname(os.path.abspath(__file__))  # AI-weather-models/Pangu/Main/
    
    # Convert relative paths to absolute paths based on script location
    if not os.path.isabs(input_dir):
        input_dir = os.path.normpath(os.path.join(script_dir, input_dir))
    if not os.path.isabs(raw_input_dir):
        raw_input_dir = os.path.normpath(os.path.join(script_dir, raw_input_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
    if not os.path.isabs(model_dir):
        model_dir = os.path.normpath(os.path.join(script_dir, model_dir))
    if not os.path.isabs(image_dir):
        image_dir = os.path.normpath(os.path.join(script_dir, image_dir))
    
    print(f"[INFO] Using paths:")
    print(f"  Raw Input:      {raw_input_dir}")
    print(f"  Processed Input: {input_dir}")
    print(f"  Output:         {output_dir}")
    print(f"  Models:         {model_dir}")
    print(f"  Images:         {image_dir}\n")
    
    workflow_start = datetime.now()
    results = {
        'status': 'running',
        'init_datetime': init_datetime_str,
        'data_source': data_source,
        'model_type': model_type,
        'start_time': workflow_start.strftime('%Y-%m-%d %H:%M:%S'),
        'data_files': [],
        'forecast_files': [],
        'image_files': [],
        'errors': []
    }
    
    print("\n" + "="*70)
    print("PANGU WEATHER FORECAST WORKFLOW")
    print("="*70)
    print(f"[START] Workflow initiated at {workflow_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set default init datetime to current time if not specified
    if init_datetime_str is None:
        init_datetime_str = datetime.now().strftime('%Y%m%d%H')
        print(f"[INFO] No init datetime specified, using current time: {init_datetime_str}")
    
    # Step 1: Data Acquisition
    print("\n" + "-"*70)
    print("STEP 1: DATA ACQUISITION")
    print("-"*70)
    
    try:
        print(f"[INFO] Acquiring {data_source} data for {init_datetime_str}...")
        
        # Create DataDownloader instance with custom directories
        downloader = DataDownloader(output_dir=input_dir, input_dir=input_dir, raw_input_dir=raw_input_dir)
        
        # Select data source and download
        if data_source.upper() == 'GFS':
            print(f"[INFO] Retrieving GFS data...")
            data_dict = downloader.get_gfs_data(init_datetime_str)
        elif data_source.upper() == 'ERA5':
            print(f"[INFO] Retrieving ERA5 data...")
            data_dict = downloader.get_era5_data(init_datetime_str)
        elif data_source.upper() == 'ECMWF':
            print(f"[INFO] Retrieving ECMWF data...")
            data_dict = downloader.get_ec_data(init_datetime_str)
        else:
            raise ValueError(f"Unknown data source: {data_source}")
        
        if data_dict is None or not data_dict:
            print(f"[ERROR] Data acquisition failed")
            results['status'] = 'failed'
            results['errors'].append('Data acquisition failed')
            return results
        
        # Find the generated NPY files
        upper_file = os.path.join(input_dir, 'input_upper.npy')
        surface_file = os.path.join(input_dir, 'input_surface.npy')
        
        if os.path.exists(upper_file):
            results['data_files'].append(upper_file)
        if os.path.exists(surface_file):
            results['data_files'].append(surface_file)
        
        print(f"[OK] Data acquisition completed successfully")
        
        # ===== 绘制初始场 =====
        if draw_results:
            print(f"\n[INFO] Plotting initial conditions...")
            try:
                from draw_pangu_results import PanguResultDrawer
                drawer = PanguResultDrawer(output_dir=image_dir)
                
                # 加载初始数据
                raw_surface = np.load(surface_file)
                raw_upper = np.load(upper_file)
                
                data_dict_plot = {
                    'mslp': raw_surface[0] / 100,  # Pa -> hPa
                    'u10': raw_surface[1],
                    'v10': raw_surface[2],
                    't2m': raw_surface[3],
                }
                
                # 绘制初始场风场和气压
                wind_file = drawer.draw_mslp_and_wind(
                    data_dict_plot, init_datetime_str, 0,
                    data_source=data_source,
                    lon_range=lon_range,
                    lat_range=lat_range
                )
                if wind_file:
                    results['image_files'].append(wind_file)
                    print(f"  [OK] Initial MSLP/wind saved: {wind_file}")
                
            except Exception as e:
                print(f"  [WARN] Failed to plot initial conditions: {e}")
        
    except Exception as e:
        print(f"[ERROR] Data acquisition error: {e}")
        results['status'] = 'failed'
        results['errors'].append(f'Data acquisition error: {e}')
        return results
    
    # Step 2: Model Execution
    print("\n" + "-"*70)
    print("STEP 2: MODEL EXECUTION (Pangu Forecast)")
    print("-"*70)
    
    try:
        print(f"[INFO] Initializing {model_type}h Pangu model...")
        
        runner = PanguRunner(
            model_type=model_type,
            input_dir=input_dir,
            output_dir=output_dir,
            model_dir=model_dir
        )
        
        print(f"[INFO] Running forecast: {run_times} x {model_type}h = {run_times * model_type}h total...")
        
        forecast_files = runner.run_forecast(
            run_times=run_times,
            datetime_str=init_datetime_str,
            data_source=data_source,
            skip_existing=skip_existing
        )
        
        if not forecast_files:
            print(f"[ERROR] Forecast execution failed")
            results['status'] = 'failed'
            results['errors'].append('Forecast execution failed')
            return results
        
        results['forecast_files'] = forecast_files
        print(f"[OK] Forecast completed successfully with {len(forecast_files)} outputs")
        
    except Exception as e:
        print(f"[ERROR] Forecast execution error: {e}")
        results['status'] = 'failed'
        results['errors'].append(f'Forecast execution error: {e}')
        return results
    
    # Step 3: Visualization
    print("\n" + "-"*70)
    print("STEP 3: VISUALIZATION")
    print("-"*70)
    
    if not draw_results:
        print("[SKIP] Visualization skipped by user")
    else:
        try:
            print(f"[INFO] Generating visualization results...")
            
            drawer = PanguResultDrawer(output_dir=image_dir)
            
            for output_file in forecast_files:
                # Extract forecast hour from filename
                filename = os.path.basename(output_file)
                try:
                    # Expected format: output_surface_YYYYMMDDHH+HHHh_SOURCE.npy
                    # Extract the part after + and before h
                    if '+' in filename:
                        forecast_str = filename.split('+')[1]  # Get '024h_SOURCE.npy'
                        forecast_hour = int(forecast_str.split('h')[0])  # Get '024' -> 24
                    else:
                        print(f"[WARN] Unable to parse forecast hour from {filename}, skipping")
                        continue
                except (IndexError, ValueError) as e:
                    print(f"[WARN] Unable to parse forecast hour from {filename}: {e}, skipping")
                    continue
                
                # Load data
                try:
                    raw_data = np.load(output_file)
                    data_dict = {
                        'mslp': raw_data[0] / 100,
                        'u10': raw_data[1],
                        'v10': raw_data[2],
                        't2m': raw_data[3],
                    }
                except Exception as e:
                    print(f"[WARN] Failed to load {filename}: {e}")
                    continue
                
                # Draw MSLP and wind
                try:
                    wind_file = drawer.draw_mslp_and_wind(
                        data_dict, init_datetime_str, forecast_hour,
                        data_source=data_source,
                        lon_range=lon_range,
                        lat_range=lat_range
                    )
                    if wind_file:
                        results['image_files'].append(wind_file)
                except Exception as e:
                    print(f"[WARN] Failed to draw MSLP/wind for +{forecast_hour:03d}h: {e}")

            if results['image_files']:
                print(f"[OK] Visualization completed successfully with {len(results['image_files'])} images")
            else:
                print(f"[WARN] No images generated")
                results['warnings'] = ['No images generated']
                
        except Exception as e:
            print(f"[ERROR] Visualization error: {e}")
            results['status'] = 'partial'
            results['errors'].append(f'Visualization error: {e}')
    
    # Workflow completion
    workflow_end = datetime.now()
    duration = (workflow_end - workflow_start).total_seconds()
    
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print(f"[OK] Workflow completed!")
    print(f"   Start time:     {workflow_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End time:       {workflow_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration:       {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"   Init datetime:  {init_datetime_str}")
    print(f"   Data source:    {data_source}")
    print(f"   Model type:     {model_type}h")
    print(f"   Forecast range: +{0:03d}h to +{run_times*model_type:03d}h")
    print(f"   Data files:     {len(results['data_files'])}")
    print(f"   Forecast files: {len(results['forecast_files'])}")
    print(f"   Image files:    {len(results['image_files'])}")
    if results['errors']:
        print(f"   Errors:         {len(results['errors'])}")
        for err in results['errors']:
            print(f"     - {err}")
    print("="*70)
    
    # 显示生成的图片位置
    if results['image_files']:
        print("\n[INFO] Generated images saved at:")
        for img_file in results['image_files']:
            print(f"  → {img_file}")
    print()
    
    results['status'] = 'completed'
    results['end_time'] = workflow_end.strftime('%Y-%m-%d %H:%M:%S')
    results['duration_seconds'] = duration
    
    return results


def main():
    """Command-line interface for workflow"""
    
    parser = argparse.ArgumentParser(
        description='Pangu Weather Forecast Complete Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main-pangu-workflow.py --data-source GFS --init-datetime 2025121200 --run-times 8
  python main-pangu-workflow.py --data-source ERA5 --run-times 4 --lon-range 100 150 --lat-range 0 50
  python main-pangu-workflow.py --data-source ECMWF --model-type 6 --skip-existing
        """
    )
    
    parser.add_argument('--data-source', type=str, default='GFS', 
                       choices=['GFS', 'ERA5', 'ECMWF'],
                       help='Data source (default: GFS)')
    parser.add_argument('--init-datetime', type=str, default=None,
                       help='Initialization datetime in format YYYYMMDDHH (default: current time)')
    parser.add_argument('--model-type', type=int, default=24, choices=[6, 24],
                       help='Model type: 6 for 6h, 24 for 24h (default: 24)')
    parser.add_argument('--run-times', type=int, default=8,
                       help='Number of forecast rounds (default: 8)')
    parser.add_argument('--lon-range', type=float, nargs=2, metavar=('LON_MIN', 'LON_MAX'),
                       help='Longitude range for visualization')
    parser.add_argument('--lat-range', type=float, nargs=2, metavar=('LAT_MIN', 'LAT_MAX'),
                       help='Latitude range for visualization')
    parser.add_argument('--no-draw', action='store_true',
                       help='Skip visualization step')
    parser.add_argument('--skip-existing', type=bool, default=True,
                       help='Skip existing data/outputs (default: True)')
    
    args = parser.parse_args()
    
    # Run workflow
    results = run_complete_workflow(
        data_source=args.data_source,
        init_datetime_str=args.init_datetime,
        model_type=args.model_type,
        run_times=args.run_times,
        lon_range=args.lon_range,
        lat_range=args.lat_range,
        draw_results=not args.no_draw,
        skip_existing=args.skip_existing
    )
    
    return results


if __name__ == '__main__':
    # Run with command-line arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Example: Run with default parameters
        print("[INFO] Running with default parameters")
        print("[INFO] Use --help to see available options\n")
        init_datetime = '2025121400'   
        #init_datetime = datetime.now().strftime('%Y%m%d%H')
        results = run_complete_workflow(
            data_source='GFS',
            init_datetime_str=init_datetime,
            model_type=24,   #1/3/6/24
            run_times=1,  
            draw_results=True,
            skip_existing=True
        )