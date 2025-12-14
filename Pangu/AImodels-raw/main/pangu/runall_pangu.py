import cdsapi
import os
import numpy as np
import onnx
import onnxruntime as ort
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from get_era5_data import get_era5_data
from get_gfs_data import get_gfs_data
from draw_mslp_and_wind import draw_mslp_and_wind
from draw_path import draw_path

def run_pangu(model_type, data_source, run_times, datetime, output_image_path):
    print("Loading ONNX model...")
    model = onnx.load(f'weight/pangu_weather_{model_type}.onnx')
    
    print("Setting ONNX Runtime options...")
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    
    print("Setting CUDA provider options...")
    cuda_provider_options = {
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': 20 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }
    
    print("Initializing ONNX Runtime session...")
    ort_session = ort.InferenceSession(
        f'weight/pangu_weather_{model_type}.onnx',
        sess_options=options,
        providers=[('CUDAExecutionProvider', cuda_provider_options)]
    )
    
    current_datetime = datetime
    output_surface_files = []
    
    for i in range(run_times+1):
        hour = model_type * i

        output_upper_filename = os.path.join(output_path, f'output_upper_{current_datetime}+{hour:02d}h_{data_source}.npy')
        output_surface_filename = os.path.join(output_path, f'output_surface_{current_datetime}+{hour:02d}h_{data_source}.npy')
        next_output_upper_filename = os.path.join(output_path, f'output_upper_{current_datetime}+{(hour+model_type):02d}h_{data_source}.npy')
        next_output_surface_filename = os.path.join(output_path, f'output_surface_{current_datetime}+{(hour+model_type):02d}h_{data_source}.npy')

        output_png_name = f'{output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{hour:02d}h.png'
        next_output_png_name = f'{output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{(hour+model_type):02d}h.png'
        output_path_png_name = f'{output_image_path}/path_{data_source}_{current_datetime}+{(hour+model_type):02d}h.png'

        if i == 0:
            print("Loading initial input data...")
            input_upper = np.load(os.path.join(input_path, 'input_upper.npy')).astype(np.float32)
            input_surface = np.load(os.path.join(input_path, 'input_surface.npy')).astype(np.float32)
            
            print("Drawing initial MSLP and wind field...")
            draw_mslp_and_wind(datetime, model_type, run_times, os.path.join(input_path, 'input_surface.npy'),
                f'{output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{hour:02d}h.png', hour, lon_min, lon_max, lat_min, lat_max, data_source)
            print(f"Saved initial image to {output_image_path}/mslp_and_wind_{data_source}_{current_datetime}+{hour:02d}h.png")

            if os.path.exists(next_output_upper_filename) and os.path.exists(next_output_surface_filename):
                print(f"Output files {next_output_upper_filename} and {next_output_surface_filename} already exist. Skipping inference.")
            else:
                print(f"Running inference from {hour:02d}h to {(hour+model_type):02d}h...")
                output, output_surface = ort_session.run(None, {'input': input_upper, 'input_surface': input_surface})
                print(f"Saving results to {next_output_upper_filename} and {next_output_surface_filename}...")
                np.save(next_output_upper_filename, output)
                np.save(next_output_surface_filename, output_surface)
                print("Drawing MSLP and wind field...")
                draw_mslp_and_wind(datetime, model_type, run_times, next_output_surface_filename, next_output_png_name, hour+model_type, lon_min, lon_max, lat_min, lat_max, data_source)
                print(f"Saved image to {next_output_png_name}")

        else:
            if os.path.exists(next_output_upper_filename) and os.path.exists(next_output_surface_filename):
                print(f"Output files {next_output_upper_filename} and {next_output_surface_filename} already exist. Skipping inference.")
            else:
                input_upper = np.load(output_upper_filename).astype(np.float32)
                input_surface = np.load(output_surface_filename).astype(np.float32)
                print(f"Running inference from {hour:02d}h to {(hour+model_type):02d}h...")
                output, output_surface = ort_session.run(None, {'input': input_upper, 'input_surface': input_surface})
                print(f"Saving results to {next_output_upper_filename} and {next_output_surface_filename}...")
                np.save(next_output_upper_filename, output)
                np.save(next_output_surface_filename, output_surface)
            print("Drawing MSLP and wind field...")
            draw_mslp_and_wind(datetime, model_type, run_times, next_output_surface_filename, next_output_png_name, hour+model_type, lon_min, lon_max, lat_min, lat_max, data_source)
            print(f"Saved image to {next_output_png_name}")
        
        output_surface_files.append(next_output_surface_filename)
        draw_path(output_surface_files, datetime, data_source, model_type, i, output_path_png_name, lon_min, lon_max, lat_min, lat_max)
        print(f"Saved image to {output_path_png_name}")
    print('Done!')

def main():
    global data_source, output_dir, output_image_path, output_path, input_path
    data_source = 'GFS'
    output_dir = 'raw_data'
    output_image_path = 'output_png\\pangu'
    output_path = 'output_data\\pangu'
    input_path = 'input'
    
    global datetime, model_type, run_times, hour, lon_min, lon_max, lat_min, lat_max
    model_type = 24
    run_times = 8
    datetime = '2025081000'
    lon_min = 100
    lon_max = 170
    lat_min = 0
    lat_max = 35

    if data_source == 'ERA5':
        get_era5_data(output_dir, datetime)
    elif data_source == 'GFS':
        get_gfs_data(output_dir, datetime)

    run_pangu(model_type, data_source, run_times, datetime, output_image_path)

if __name__ == '__main__':
    main()