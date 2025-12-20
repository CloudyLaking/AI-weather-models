# AI-weather-models\AIFS\Longterm_Experiment\longterm_integration.py
# AIFS 长期自由积分实验：沿用 Main 工作流逻辑执行长时间积分

import os
import sys
import numpy as np
from datetime import datetime

# 注入 Main 目录，直接复用 get_data_aifs、run_aifs、draw_aifs_results
project_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_parent, 'Main'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from get_data_aifs import AIFSDataDownloader
    from run_aifs import AIFSRunner
    from draw_aifs_results import AIFSResultDrawer
except ImportError as exc:
    print(f"[ERROR] Failed to import AIFS modules: {exc}")
    sys.exit(1)


# ==========================================
# 全局配置
# ==========================================
START_YEAR = 1990
START_MONTH = 7
START_DAY = 1
END_YEAR = 2020
LEAD_TIME_HOURS = 6
EXPECTED_FIELD_POINTS = 542080


# ==========================================
# 工具函数（与 Main 工作流保持一致的风格）
# ==========================================
def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(current, 'Models-weights')):
            return current
        current = os.path.dirname(current)
    return os.getcwd()


def setup_directories(project_root, start_year):
    exp_name = f'Longterm_{start_year}'
    data_disk = os.path.join(project_root, 'autodl-tmp')
    base = data_disk if os.path.exists(data_disk) else project_root
    if base == data_disk:
        print(f"[INFO] Using data disk: {data_disk}")
    else:
        print("[WARN] Data disk not found, falling back to project root")

    dirs = {
        'input': os.path.join(base, 'Input', 'AIFS', exp_name),
        'raw_input': os.path.join(base, 'Input', 'AIFS_raw', exp_name),
        'output_daily': os.path.join(base, 'Output', 'AIFS', exp_name, 'Daily'),
        'output_full': os.path.join(base, 'Output', 'AIFS', exp_name, 'Full'),
        'images': os.path.join(base, 'Run-output-png', 'AIFS', exp_name, 'ERA5')
    }

    for label, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directory ready: {label} -> {path}")

    return dirs


def ensure_lat_lon(state, *fallback_states):
    for key in ('latitudes', 'longitudes'):
        arr = state.get(key)
        if isinstance(arr, np.ndarray) and arr.size:
            continue
        for fallback in fallback_states:
            if fallback is None:
                continue
            candidate = fallback.get(key)
            if isinstance(candidate, np.ndarray) and candidate.size:
                state[key] = candidate
                break


def extract_fields_from_state(state):
    fields = state.get('fields', {})
    surface = {
        'mslp': fields.get('msl', np.array([])),
        'u10': fields.get('10u', np.array([])),
        'v10': fields.get('10v', np.array([])),
        't2m': fields.get('2t', np.array([]))
    }
    upper = {}
    for level in [850, 500, 200]:
        for var in ['z', 't', 'u', 'v']:
            key = f"{var}_{level}"
            upper[key] = fields.get(key, np.array([]))
    return surface, upper


def save_daily_data(surface_fields, upper_fields, output_path, current_date, forecast_hours=None):
    date_str = current_date.strftime('%Y%m%d')
    suffix = f"_+{forecast_hours:04d}h" if forecast_hours is not None else ''
    surface_file = os.path.join(output_path, f"daily_surface_{date_str}{suffix}.npz")
    upper_file = os.path.join(output_path, f"daily_upper_{date_str}{suffix}.npz")
    np.savez_compressed(surface_file, **surface_fields)
    np.savez_compressed(upper_file, **upper_fields)
    return surface_file, upper_file


def save_full_data(state, output_path, current_date, forecast_hours=None):
    date_str = current_date.strftime('%Y%m%d')
    suffix = f"_+{forecast_hours:04d}h" if forecast_hours is not None else ''
    filepath = os.path.join(output_path, f"full_state_{date_str}{suffix}.npz")
    payload = {
        'date': state['date'].isoformat(),
        'latitudes': state.get('latitudes', np.array([])),
        'longitudes': state.get('longitudes', np.array([]))
    }
    for name, value in state.get('fields', {}).items():
        if isinstance(value, np.ndarray):
            payload[name] = value
    np.savez_compressed(filepath, **payload)
    return filepath


def should_save_full(current_date):
    return ((current_date.month, current_date.day) == (1, 1)) or \
           ((current_date.month, current_date.day) == (7, 1))


def describe_field_shapes(state):
    sample = None
    for value in state.get('fields', {}).values():
        sample = np.asarray(value)
        break
    if sample is not None:
        print(f"  Field shape example: {sample.shape} (expected: (2, {EXPECTED_FIELD_POINTS}))")


def build_next_input_state(prev_input_state, forecast_state):
    prev_fields = prev_input_state.get('fields', {})
    forecast_fields = forecast_state.get('fields', {})
    next_fields = {}
    reused = []

    for name, prev_data in prev_fields.items():
        prev_arr = np.asarray(prev_data)
        if prev_arr.ndim == 1:
            prev_vector = prev_arr.reshape(-1)
        else:
            prev_vector = prev_arr[-1].reshape(-1)

        forecast_data = forecast_fields.get(name)
        if forecast_data is None:
            reused.append(name)
            forecast_vector = prev_vector
        else:
            forecast_arr = np.asarray(forecast_data)
            if forecast_arr.ndim == 1:
                forecast_vector = forecast_arr.reshape(-1)
            else:
                forecast_vector = forecast_arr[-1].reshape(-1)

        if prev_vector.shape != forecast_vector.shape:
            print(f"[WARN] Shape mismatch for {name}: prev={prev_vector.shape}, curr={forecast_vector.shape}")
            continue

        next_fields[name] = np.stack([prev_vector, forecast_vector], axis=0)

    if reused:
        print(f"[INFO] Carrying static forcings forward: {sorted(reused)}")

    if not next_fields:
        print("[ERROR] Unable to build next input: no overlapping fields")
        return None

    next_state = {
        'date': forecast_state['date'],
        'latitudes': forecast_state.get('latitudes'),
        'longitudes': forecast_state.get('longitudes'),
        'fields': next_fields
    }
    ensure_lat_lon(next_state, prev_input_state, forecast_state)
    return next_state


def log_progress(current_date, start_time):
    elapsed = int((current_date - start_time).total_seconds() / 3600)
    print(f"[PROGRESS] {current_date.strftime('%Y-%m-%d %H:%M')} (+{elapsed}h)")


def print_step(title):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


# ==========================================
# 长期积分主流程
# ==========================================
def run_longterm_integration(start_year, end_year=2020, start_month=7, start_day=1):
    project_root = find_project_root()
    print(f"[INFO] Project root detected as: {project_root}")

    dirs = setup_directories(project_root, start_year)

    start_date = datetime(start_year, start_month, start_day, 0)
    end_date = datetime(end_year, 12, 31, 0)
    current_date = start_date

    print("\n" + "=" * 70)
    print("AIFS LONG-TERM INTEGRATION")
    print("=" * 70)
    print(f"Start date: {start_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"End date:   {end_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"Model step: {LEAD_TIME_HOURS} h")
    print("=" * 70)

    # STEP 1: 数据获取（ERA5）
    print_step("STEP 1: DATA ACQUISITION (ERA5)")

    downloader = AIFSDataDownloader(
        input_dir=dirs['input'],
        raw_input_dir=dirs['raw_input']
    )

    init_str = current_date.strftime('%Y%m%d%H')
    print(f"[INFO] Acquiring ERA5 initial state: {init_str}")
    input_state = downloader.process_era5_data(init_str)
    if not input_state:
        print("[ERROR] Failed to retrieve initial ERA5 state")
        return

    ensure_lat_lon(input_state)
    print(f"[OK] Initial conditions loaded ({len(input_state.get('fields', {}))} fields)")
    describe_field_shapes(input_state)

    surface_fields, upper_fields = extract_fields_from_state(input_state)
    save_daily_data(surface_fields, upper_fields, dirs['output_daily'], current_date)
    save_full_data(input_state, dirs['output_full'], current_date)
    print(f"[SAVE] Initial ERA5 fields archived for {current_date.date()}")

    try:
        init_drawer = AIFSResultDrawer(output_dir=dirs['images'])
        input_state['init_date'] = input_state['date']
        init_img = init_drawer.draw_mslp_and_wind(
            input_state,
            init_datetime_str=init_str,
            data_source='ERA5',
            lon_range=[0, 360],
            lat_range=[-90, 90]
        )
        if init_img:
            print(f"  [OK] Initial plot saved: {os.path.basename(init_img)}")
    except Exception as exc:
        print(f"  [WARN] Failed to draw initial image: {exc}")

    # STEP 2: 初始化 Runner
    print_step("STEP 2: MODEL INITIALIZATION")
    runner = AIFSRunner(device='cuda', output_dir=dirs['output_daily'], use_huggingface=False)
    if runner.runner is None:
        print("[ERROR] AIFS Runner initialization failed")
        return
    print("[OK] AIFS Runner ready")

    drawer = AIFSResultDrawer(output_dir=dirs['images'])

    # STEP 3: 长期积分
    print_step("STEP 3: LONG-TERM INTEGRATION")

    step_count = 0
    day_count = 0

    while current_date <= end_date:
        print(f"\n[RUN] Forecast starting from {current_date.strftime('%Y-%m-%d %H:%M')}")
        latest_state = None

        try:
            for state in runner.runner.run(input_state=input_state, lead_time=LEAD_TIME_HOURS):
                latest_state = state
        except Exception as exc:
            print(f"[ERROR] Runner failed at {current_date}: {exc}")
            import traceback
            traceback.print_exc()
            break

        if latest_state is None:
            print("[ERROR] Runner returned no states; aborting")
            break

        current_date = latest_state['date']
        step_count += 1
        log_progress(current_date, start_date)

        hours_since_start = int((current_date - start_date).total_seconds() / 3600)

        if current_date.hour == 0:
            day_count += 1
            surface_fields, upper_fields = extract_fields_from_state(latest_state)
            save_daily_data(surface_fields, upper_fields, dirs['output_daily'], current_date,
                            forecast_hours=hours_since_start)

            if should_save_full(current_date):
                save_full_data(latest_state, dirs['output_full'], current_date,
                               forecast_hours=hours_since_start)
                try:
                    latest_state['init_date'] = start_date
                    image_file = drawer.draw_mslp_and_wind(
                        latest_state,
                        init_datetime_str=start_date.strftime('%Y%m%d%H'),
                        data_source='AIFS',
                        lon_range=[0, 360],
                        lat_range=[-90, 90]
                    )
                    if image_file:
                        print(f"  [OK] Synoptic image saved: {os.path.basename(image_file)}")
                except Exception as exc:
                    print(f"  [WARN] Failed to draw synoptic image: {exc}")

        next_input = build_next_input_state(input_state, latest_state)
        if next_input is None:
            print("[ERROR] Cannot build next input state; stopping integration")
            break

        input_state = next_input

    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETED")
    print("=" * 70)
    print(f"Total steps: {step_count}")
    print(f"Total days: {day_count}")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    print("[INFO] Starting long-term integration run...")
    run_longterm_integration(START_YEAR, END_YEAR, START_MONTH, START_DAY)