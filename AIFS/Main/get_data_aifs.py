# AI-weather-models\AIFS\Main\get_data_aifs.py
# AIFS 数据下载和转换 - 从 ECMWF Open Data 或 ERA5 获取初始条件并转换为 NPY 格式

import os
import sys
import numpy as np
import atexit

# 抑制临时文件清理错误（Windows特定）
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

# 在其他第三方库（特别是 earthkit）之前先导入 requests 并打补丁
import requests

# ---- Windows 下修正 earthkit 矩阵下载 URL 中的 '\' / '%5C' ----
from requests.sessions import Session as _Session
_original_request = _Session.request

def _patched_request(self, method, url, *args, **kwargs):
    # 只针对 earthkit regrid 的矩阵 URL 做修正
    if "earthkit/regrid/db" in url and ("\\" in url or "%5C" in url):
        fixed_url = url.replace("\\", "/").replace("%5C", "/")
        if fixed_url != url:
            print(f"[PATCH] Fix earthkit matrix URL:\n  {url}\n  -> {fixed_url}")
            url = fixed_url
    return _original_request(self, method, url, *args, **kwargs)

_Session.request = _patched_request
# ---- 补丁结束 ----

from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

try:
    import earthkit.data as ekd
    import earthkit.regrid as ekr
except ImportError:
    print("[ERROR] earthkit package not installed, run: pip install earthkit-data earthkit-regrid")
    sys.exit(1)

try:
    from ecmwf.opendata import Client as OpendataClient
except ImportError:
    print("[ERROR] ecmwf-opendata package not installed, run: pip install ecmwf-opendata")
    sys.exit(1)

try:
    import cdsapi
except ImportError:
    print("[WARN] cdsapi not installed, ERA5 support disabled. Run: pip install cdsapi")
    cdsapi = None


class AIFSDataDownloader:
    """AIFS 数据下载和转换类 - 从 ECMWF Open Data 获取数据并转换为 NPY 格式"""
    
    # AIFS 支持的参数列表
    PARAM_SFC = ["10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "lsm", "z", "slor", "sdor"]
    PARAM_SOIL = ["vsw", "sot"]
    PARAM_PL = ["gh", "t", "u", "v", "w", "q"]
    LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
    SOIL_LEVELS = [1, 2]
    
    # 土壤参数重映射（AIFS模型训练时使用的原始名称）
    SOIL_MAPPING = {
        'sot_1': 'stl1',
        'sot_2': 'stl2',
        'vsw_1': 'swvl1',
        'vsw_2': 'swvl2'
    }
    
    @staticmethod
    def print_input_stats(input_dir):
        """
        打印已生成的 input_*.npz 文件中各变量的统计信息
        
        Args:
            input_dir: 输入数据目录
        """
        try:
            # 查找所有 .npz 文件
            import glob
            npz_files = glob.glob(os.path.join(input_dir, 'input_*.npz'))
            
            for npz_file in npz_files:
                data = np.load(npz_file, allow_pickle=True)
                print(f"[INPUT] {os.path.basename(npz_file)}:")
                for key in data.files:
                    if key != 'date':
                        arr = data[key]
                        print(f"  ✓ {key}: min={arr.min():.4f}, max={arr.max():.4f}")
        except Exception as e:
            pass  # 如果出错就不打印，不影响程序运行
    
    def __init__(self, input_dir='Input/AIFS', 
                 raw_input_dir='Input/AIFS_raw'):
        """
        初始化数据下载器
        
        参数:
            input_dir: 处理后数据存储目录（NPY 文件）
            raw_input_dir: 原始数据存储目录（NetCDF 文件）
        """
        # 如果是相对路径，则基于当前脚本所在目录转换为绝对路径
        if not os.path.isabs(input_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            input_dir = os.path.normpath(os.path.join(script_dir, input_dir))
        
        if not os.path.isabs(raw_input_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            raw_input_dir = os.path.normpath(os.path.join(script_dir, raw_input_dir))
        
        self.input_dir = input_dir
        self.raw_input_dir = raw_input_dir
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.raw_input_dir, exist_ok=True)
        
        # 设置earthkit缓存目录，使用前向斜杠以解决Windows路径问题
        # 这防止反斜杠被包含在URL中导致404错误
        cache_dir = os.path.join(self.raw_input_dir, '.earthkit_cache')
        os.makedirs(cache_dir, exist_ok=True)
        # 使用前向斜杠替换反斜杠，防止URL编码问题
        cache_dir_posix = cache_dir.replace('\\', '/')
        os.environ['EARTHKIT_CACHE_DIR'] = cache_dir_posix
        
        print(f"[INFO] Input directory: {self.input_dir}")
        print(f"[INFO] Raw input directory: {self.raw_input_dir}")
        print(f"[INFO] EarthKit cache: {cache_dir_posix}")
    
    @staticmethod
    def get_latest_date():
        """
        获取最新可用的 ECMWF Open Data 日期
        
        返回:
            datetime: 最新数据日期
        """
        try:
            client = OpendataClient()
            latest_date = client.latest()
            print(f"[OK] Latest ECMWF Open Data date: {latest_date}")
            return latest_date
        except Exception as e:
            print(f"[ERROR] Failed to get latest date: {e}")
            return None
    
    def get_open_data(self, param, date, levelist=None, save_raw=True, raw_file_prefix=None):
        """
        从 ECMWF Open Data 获取数据并插值到 N320 分辨率
        
        参数:
            param: 参数列表
            date: 日期
            levelist: 气压层列表（对于气压层参数）
            save_raw: 是否保存原始 GRIB 文件
            raw_file_prefix: 原始文件前缀（用于区分 surface/soil/pressure）
            
        返回:
            dict: 参数名称 -> 数据数组的字典
        """
        fields = defaultdict(list)
        levelist = levelist or []
        
        try:
            # 获取当前日期和前6小时的数据
            for offset_hours in [6, 0]:
                current_date = date - timedelta(hours=offset_hours)
                print(f"  [LOAD] Retrieving data for {current_date}...")
                
                try:
                    data = ekd.from_source(
                        "ecmwf-open-data",
                        date=current_date,
                        param=param,
                        levelist=levelist
                    )
                    
                    # 保存原始 GRIB 文件
                    if save_raw and raw_file_prefix:
                        date_str = current_date.strftime('%Y%m%d%H')
                        raw_filename = f"{raw_file_prefix}_{date_str}.grib2"
                        raw_filepath = os.path.join(self.raw_input_dir, raw_filename)
                        
                        if not os.path.exists(raw_filepath):
                            try:
                                data.save(raw_filepath)
                                print(f"  [SAVE] Raw GRIB saved: {raw_filename}")
                            except Exception as e:
                                print(f"  [WARN] Failed to save raw GRIB: {e}")
                    
                except Exception as e:
                    print(f"  [WARN] Failed to retrieve data for {current_date}: {e}")
                    continue
                
                for f in data:
                    # Open data 经度范围是 -180 到 180，需要转换到 0-360
                    values = f.to_numpy()
                    assert values.shape == (721, 1440), f"Unexpected shape: {values.shape}"
                    
                    # 经度转换
                    values = np.roll(values, -values.shape[1] // 2, axis=1)
                    
                    # 插值到 N320 分辨率（64x128）
                    print(f"    [INTERP] Interpolating {f.metadata('param')} to N320...")
                    values = ekr.interpolate(
                        values,
                        {"grid": (0.25, 0.25)},  # 源网格分辨率
                        {"grid": "N320"}  # 目标网格分辨率
                    )
                    
                    # 打印数据统计信息
                    print(f"      [STATS] {f.metadata('param')}: min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
                    
                    # 构建参数名称
                    if levelist:
                        param_name = f"{f.metadata('param')}_{f.metadata('levelist')}"
                    else:
                        param_name = f.metadata("param")
                    
                    fields[param_name].append(values)
            
            # 将各参数的时间序列堆叠为单一矩阵
            for param_name, values in fields.items():
                if len(values) > 0:
                    stacked = np.stack(values)
                    print(f"  [FINAL] {param_name}: shape={stacked.shape}, min={stacked.min():.4f}, max={stacked.max():.4f}, mean={stacked.mean():.4f}")
                    fields[param_name] = stacked
                else:
                    fields[param_name] = None
            
            # 移除None值
            fields = {k: v for k, v in fields.items() if v is not None}
            
            return dict(fields)
            
        except Exception as e:
            print(f"[ERROR] Data retrieval error: {e}")
            return {}
    
    def process_era5_data(self, datetime_str, skip_existing=True):
        """
        从 ERA5 重分析数据获取并处理 AIFS 初始条件
        
        参数:
            datetime_str: 日期时间字符串，格式 'YYYYMMDDHH'
            skip_existing: 是否跳过已存在的文件
            
        返回:
            dict: 处理后的字段数据
        """
        if cdsapi is None:
            print("[ERROR] CDS API not available, run: pip install cdsapi")
            return None
        
        print(f"\n[START] Starting AIFS data acquisition from ERA5")
        
        # 解析日期时间
        try:
            date = datetime.strptime(datetime_str, '%Y%m%d%H')
            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            hour = date.strftime('%H')
        except ValueError as e:
            print(f"[ERROR] Invalid datetime format: {e}")
            return None
        
        # 检查缓存
        npy_file = os.path.join(self.input_dir, f'input_era5_{datetime_str}.npy')
        if skip_existing and os.path.exists(npy_file):
            print(f"[SKIP] Processed data already exists: {npy_file}")
            return self.load_input_state(npy_file)
        
        try:
            client = cdsapi.Client()
            
            fields = {}
            
            # 下载单层参数
            print(f"\n[STEP 1/3] Downloading ERA5 single-level parameters...")
            single_level_file = os.path.join(self.raw_input_dir, f'era5_single_{datetime_str}.nc')
            
            if not os.path.exists(single_level_file):
                request = {
                    'product_type': 'reanalysis',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind',
                        '2m_dewpoint_temperature', '2m_temperature',
                        'mean_sea_level_pressure', 'skin_temperature',
                        'surface_pressure', 'total_column_water',
                        'land_sea_mask', 'geopotential',
                        'slope_of_topography', 'standard_deviation_of_topography'
                    ],
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': hour,
                    'format': 'netcdf',
                }
                
                print(f"  [DOWNLOAD] Fetching from CDS...")
                client.retrieve('reanalysis-era5-single-levels', request, single_level_file)
                print(f"  [OK] Downloaded: {single_level_file}")
            
            # 读取并处理单层数据
            print(f"  [PROCESS] Processing single-level data...")
            ds_single = xr.open_dataset(single_level_file)
            
            # 参数映射（ERA5变量名 -> AIFS字段名）
            mapping = {
                '10m_u_component_of_wind': '10u',
                '10m_v_component_of_wind': '10v',
                '2m_dewpoint_temperature': '2d',
                '2m_temperature': '2t',
                'mean_sea_level_pressure': 'msl',
                'skin_temperature': 'skt',
                'surface_pressure': 'sp',
                'total_column_water': 'tcw',
                'land_sea_mask': 'lsm',
                'geopotential': 'z',
                'slope_of_topography': 'slor',
                'standard_deviation_of_topography': 'sdor',
            }
            
            for era5_var, aifs_var in mapping.items():
                if era5_var in ds_single.data_vars:
                    # 提取并扩展为2个时间步（模拟与前6小时数据）
                    data = ds_single[era5_var].values
                    # 复制为两个时间步
                    fields[aifs_var] = np.stack([data, data])
                    print(f"    [OK] {aifs_var}")
            
            ds_single.close()
            print(f"  [OK] Retrieved {len(fields)} single-level parameters")
            
            # 下载气压层参数
            print(f"\n[STEP 2/3] Downloading ERA5 pressure level parameters...")
            pressure_file = os.path.join(self.raw_input_dir, f'era5_pressure_{datetime_str}.nc')
            
            if not os.path.exists(pressure_file):
                request = {
                    'product_type': 'reanalysis',
                    'variable': [
                        'geopotential', 'temperature',
                        'u_component_of_wind', 'v_component_of_wind',
                        'specific_humidity', 'vertical_velocity'
                    ],
                    'pressure_level': [
                        '1000', '925', '850', '700', '600', '500',
                        '400', '300', '250', '200', '150', '100', '50'
                    ],
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': hour,
                    'format': 'netcdf',
                }
                
                print(f"  [DOWNLOAD] Fetching from CDS...")
                client.retrieve('reanalysis-era5-pressure-levels', request, pressure_file)
                print(f"  [OK] Downloaded: {pressure_file}")
            
            # 读取并处理气压层数据
            print(f"  [PROCESS] Processing pressure level data...")
            ds_pressure = xr.open_dataset(pressure_file)
            
            # 提取气压层数据
            pressure_vars = {
                'geopotential': 'z',
                'temperature': 't',
                'u_component_of_wind': 'u',
                'v_component_of_wind': 'v',
                'specific_humidity': 'q',
                'vertical_velocity': 'w',
            }
            
            pl_count = 0
            for era5_var, aifs_base in pressure_vars.items():
                if era5_var in ds_pressure.data_vars:
                    data = ds_pressure[era5_var].values  # Shape: (levels, lat, lon)
                    
                    # 逐个压力层处理
                    for i, level in enumerate(self.LEVELS):
                        field_name = f"{aifs_base}_{level}"
                        # 扩展为2个时间步
                        fields[field_name] = np.stack([data[i], data[i]])
                        pl_count += 1
            
            ds_pressure.close()
            print(f"  [OK] Retrieved {pl_count} pressure level parameters")
            
            # 处理土壤参数（如果可用）
            print(f"\n[STEP 3/3] Processing soil parameters...")
            soil_file = os.path.join(self.raw_input_dir, f'era5_soil_{datetime_str}.nc')
            
            if not os.path.exists(soil_file):
                request = {
                    'product_type': 'reanalysis',
                    'variable': [
                        'soil_temperature_level_1', 'soil_temperature_level_2',
                        'volumetric_soil_water_level_1', 'volumetric_soil_water_level_2',
                    ],
                    'year': year,
                    'month': month,
                    'day': day,
                    'time': hour,
                    'format': 'netcdf',
                }
                
                try:
                    print(f"  [DOWNLOAD] Fetching soil data...")
                    client.retrieve('reanalysis-era5-land', request, soil_file)
                    print(f"  [OK] Downloaded: {soil_file}")
                except Exception as e:
                    print(f"  [WARN] Soil data unavailable: {e}")
                    soil_file = None
            
            if soil_file and os.path.exists(soil_file):
                ds_soil = xr.open_dataset(soil_file)
                
                soil_mapping = {
                    'soil_temperature_level_1': 'stl1',
                    'soil_temperature_level_2': 'stl2',
                    'volumetric_soil_water_level_1': 'swvl1',
                    'volumetric_soil_water_level_2': 'swvl2',
                }
                
                for era5_var, aifs_var in soil_mapping.items():
                    if era5_var in ds_soil.data_vars:
                        data = ds_soil[era5_var].values
                        fields[aifs_var] = np.stack([data, data])
                        print(f"    [OK] {aifs_var}")
                
                ds_soil.close()
            
            # 创建输入状态
            input_state = {
                'date': date,
                'fields': fields
            }
            
            print(f"\n[OK] Total fields retrieved: {len(fields)}")
            
            # 保存为缓存
            self.save_input_state(input_state, npy_file)
            
            return input_state
            
        except Exception as e:
            print(f"[ERROR] ERA5 data processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_ecmwf_data(self, datetime_str=None, skip_existing=True):
        """
        从 ECMWF Open Data 获取并处理 AIFS 初始条件
        
        参数:
            datetime_str: 指定日期时间字符串，格式 'YYYYMMDDHH'
            skip_existing: 是否跳过已存在的文件
            
        返回:
            dict: 处理后的字段数据
        """
        # 设置earthkit缓存，使用前向斜杠以解决Windows URL编码问题
        cache_dir = os.path.join(self.raw_input_dir, '.earthkit_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_dir_posix = cache_dir.replace('\\', '/')
        os.environ['EARTHKIT_CACHE_DIR'] = cache_dir_posix
        
        print(f"\n[START] Starting AIFS data acquisition from ECMWF Open Data")
        
        # 确定目标日期
        if datetime_str:
            try:
                date = datetime.strptime(datetime_str, '%Y%m%d%H')
            except ValueError as e:
                print(f"[ERROR] Invalid datetime format: {e}")
                return None
        else:
            date = self.get_latest_date()
            if date is None:
                print("[ERROR] Could not determine date")
                return None
        
        print(f"[INFO] Using date: {date}")
        
        # 检查是否已存在处理后的数据
        npy_file = os.path.join(self.input_dir, 'input_aifs.npy')
        if skip_existing and os.path.exists(npy_file):
            print(f"[SKIP] Processed data already exists: {npy_file}")
            return self.load_input_state(npy_file)
        
        try:
            fields = {}
            
            # 第1步：获取地面参数
            print(f"\n[STEP 1/3] Retrieving surface parameters...")
            sfc_fields = self.get_open_data(
                self.PARAM_SFC, date, 
                save_raw=True, 
                raw_file_prefix='ecmwf_surface'
            )
            fields.update(sfc_fields)
            print(f"  [OK] Retrieved {len(sfc_fields)} surface parameters")
            for name, data in sfc_fields.items():
                if data is not None:
                    print(f"    └─ {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
            
            # 第2步：获取土壤参数
            print(f"\n[STEP 2/3] Retrieving soil parameters...")
            soil_fields = self.get_open_data(
                self.PARAM_SOIL, date, 
                levelist=self.SOIL_LEVELS,
                save_raw=True,
                raw_file_prefix='ecmwf_soil'
            )
            
            # 重映射土壤参数名称
            for old_name, new_name in self.SOIL_MAPPING.items():
                if old_name in soil_fields:
                    data = soil_fields[old_name]
                    fields[new_name] = data
                    print(f"  ✓ {old_name} → {new_name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
            
            print(f"  [OK] Retrieved and remapped {len(self.SOIL_MAPPING)} soil parameters")
            
            # 第3步：获取气压层参数
            print(f"\n[STEP 3/3] Retrieving pressure level parameters...")
            pl_fields = self.get_open_data(
                self.PARAM_PL, date, 
                levelist=self.LEVELS,
                save_raw=True,
                raw_file_prefix='ecmwf_pressure'
            )
            
            # 地位势高度 (GH) 转换为地位势 (Z)
            # GH 单位为 m²/s²，Z = GH * 9.80665
            print(f"  [CONVERT] Converting geopotential height to geopotential...")
            for level in self.LEVELS:
                gh_key = f"gh_{level}"
                z_key = f"z_{level}"
                if gh_key in pl_fields:
                    gh_data = pl_fields.pop(gh_key)
                    z_data = gh_data * 9.80665
                    pl_fields[z_key] = z_data
                    print(f"    └─ {gh_key} → {z_key}: min={z_data.min():.4f}, max={z_data.max():.4f}")
            
            fields.update(pl_fields)
            print(f"  [OK] Retrieved {len(pl_fields)} pressure level parameters")
            for name, data in pl_fields.items():
                if data is not None:
                    print(f"    └─ {name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
            
            print(f"\n[OK] Total fields retrieved: {len(fields)}")
            
            # 创建初始状态字典
            input_state = {
                'date': date,
                'fields': fields
            }
            
            # 保存为 NPY 格式（用于后续快速加载）
            self.save_input_state(input_state, npy_file)
            
            return input_state
            
        except Exception as e:
            print(f"[ERROR] Data processing error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_input_state(self, input_state, npy_file):
        """
        保存输入状态为 NPY 文件（用于缓存）
        
        参数:
            input_state: 输入状态字典
            npy_file: 输出文件路径
        """
        try:
            # 创建可序列化的版本
            state_data = {
                'date': input_state['date'].isoformat(),
                'fields': {k: v for k, v in input_state['fields'].items()}
            }
            
            # 使用 NumPy 的目录格式保存（支持多个字段）
            npz_file = npy_file.replace('.npy', '.npz')
            np.savez_compressed(
                npz_file,
                **state_data
            )
            print(f"[OK] Input state cached: {npz_file}")
            self.print_input_stats(os.path.dirname(npz_file))
        except Exception as e:
            print(f"[WARN] Failed to cache input state: {e}")
    
    def load_input_state(self, npy_file):
        """
        从缓存加载输入状态
        
        参数:
            npy_file: 缓存文件路径
            
        返回:
            dict: 输入状态字典或 None
        """
        try:
            npz_file = npy_file.replace('.npy', '.npz')
            if not os.path.exists(npz_file):
                return None
            
            data = np.load(npz_file, allow_pickle=True)
            date = datetime.fromisoformat(str(data['date']))
            fields = {k: v for k, v in data.items() if k != 'date'}
            
            return {
                'date': date,
                'fields': fields
            }
        except Exception as e:
            print(f"[WARN] Failed to load cached input state: {e}")
            return None


def get_aifs_data(datetime_str=None, data_source='ECMWF', skip_existing=True):
    """
    便捷函数：一键获取 AIFS 初始条件
    
    参数:
        datetime_str: 日期时间字符串，格式 'YYYYMMDDHH'
        data_source: 数据源 'ECMWF'（最新数据）或 'ERA5'（历史数据）
        skip_existing: 是否跳过已存在的文件
        
    返回:
        dict: 输入状态字典（包含日期和字段数据）
    """
    downloader = AIFSDataDownloader()
    
    if data_source.upper() == 'ERA5':
        if datetime_str is None:
            print("[ERROR] ERA5 requires explicit datetime (YYYYMMDDHH)")
            return None
        return downloader.process_era5_data(datetime_str, skip_existing)
    elif data_source.upper() in ['ECMWF', 'OPEN-DATA']:
        return downloader.process_ecmwf_data(datetime_str, skip_existing)
    else:
        print(f"[ERROR] Unknown data source: {data_source}")
        return None


if __name__ == '__main__':
    # 示例用法
    input_state = get_aifs_data(datetime_str='2023040100', data_source='ERA5', skip_existing=True)
    
    if input_state:
        print(f"\n[INFO] Input state retrieved successfully")
        print(f"  Date: {input_state['date']}")
        print(f"  Fields: {list(input_state['fields'].keys())}")
    else:
        print("[ERROR] Failed to retrieve input state")
