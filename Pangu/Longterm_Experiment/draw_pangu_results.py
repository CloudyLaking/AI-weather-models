# AI-weather-models\Pangu\Main\draw-pangu-results.py
# 可视化 Pangu 天气预报结果（生成专业气象图）

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import sys


class PanguResultDrawer:
    """Pangu 天气预报结果绘图类"""

    def __init__(self, output_dir='Run-output-png/Pangu', figsize=(16, 10)):
        """
        初始化绘图器

        参数:
            output_dir: PNG 图片输出目录（相对或绝对路径）
            figsize: 图像尺寸 (宽, 高)
        """
        # 如果是相对路径，则基于当前脚本所在目录转换为绝对路径
        if not os.path.isabs(output_dir):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.normpath(os.path.join(script_dir, output_dir))

        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _get_output_path(self, filename, data_source='GFS'):
        """
        根据数据源获取输出路径，自动在数据源子文件夹中保存
        
        参数:
            filename: 图片文件名
            data_source: 数据源名称 (GFS, ERA5, ECMWF等)
            
        返回:
            str: 完整的输出文件路径
        """
        source_dir = os.path.join(self.output_dir, data_source.upper())
        os.makedirs(source_dir, exist_ok=True)
        return os.path.join(source_dir, filename)

    def draw_mslp_and_wind(self, data_dict, init_datetime_str, forecast_hour,
                           data_source='GFS', lon_range=None, lat_range=None):
        """
        绘制海平面气压（MSLP）等压线与 10m 风场

        参数:
            data_dict: 包含 'mslp', 'u10', 'v10' 的数据字典
            init_datetime_str: 起报时间字符串 'YYYYMMDDHH'
            forecast_hour: 预报时效（小时）
            data_source: 数据源标记（如 GFS / ERA5 等）
            lon_range: 经度范围 [lon_min, lon_max]，默认 [95, 150]
            lat_range: 纬度范围 [lat_min, lat_max]，默认 [5, 35]

        返回:
            str: 输出图片文件完整路径
        """
        if not all(k in data_dict for k in ['mslp', 'u10', 'v10']):
            print("[ERROR] data_dict 必须包含键 'mslp', 'u10', 'v10'")
            return None

        # 若未指定范围，使用东亚默认范围
        if lon_range is None or lat_range is None:
            lon_range = [95, 150]
            lat_range = [5, 35]

        mslp = data_dict['mslp']  # 形状: (721, 1440)
        u10 = data_dict['u10']
        v10 = data_dict['v10']
        wind_speed = np.sqrt(u10**2 + v10**2)

        # 构建全球经纬度网格
        lat = np.linspace(90, -90, mslp.shape[0])
        lon = np.linspace(0, 360, mslp.shape[1])
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # 提取指定经纬度范围
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        lon_indices = np.where((lon_grid[0, :] >= lon_min) & (lon_grid[0, :] <= lon_max))[0]
        lat_indices = np.where((lat_grid[:, 0] >= lat_min) & (lat_grid[:, 0] <= lat_max))[0]

        mslp = mslp[np.ix_(lat_indices, lon_indices)]
        u10 = u10[np.ix_(lat_indices, lon_indices)]
        v10 = v10[np.ix_(lat_indices, lon_indices)]
        wind_speed = wind_speed[np.ix_(lat_indices, lon_indices)]
        lon_grid = lon_grid[np.ix_(lat_indices, lon_indices)]
        lat_grid = lat_grid[np.ix_(lat_indices, lon_indices)]

        # 对气压场做高斯平滑
        mslp_smooth = gaussian_filter(mslp, sigma=2)

        # 子采样风矢量（每 5 个格点取一个，保证图上不太密）
        skip = (slice(None, None, 5), slice(None, None, 5))
        u10_skip = u10[skip]
        v10_skip = v10[skip]
        lon_skip = lon_grid[skip]
        lat_skip = lat_grid[skip]

        # 统计量
        max_wind_speed = np.max(wind_speed)
        min_pressure = np.min(mslp)

        # 创建图像与地图坐标轴
        fig = plt.figure(figsize=(15, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())

        # 自定义风速分级及颜色
        wind_speed_color_map = {
            0: '#FFFFFFFF', 5: '#D1D1D1FF', 10: '#188FD8FF', 15: '#34D259FF',
            20: '#F1B04DFF', 25: '#FF6200FF', 30: '#FF0000FF', 35: '#9C2EBAFF',
            40: '#FF00FFFF', 45: '#EA00FFFF'
        }
        wind_speed_levels = list(wind_speed_color_map.keys())
        wind_speed_colors = list(wind_speed_color_map.values())

        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            'wind_speed_cmap',
            list(zip(np.linspace(0, 1, len(wind_speed_levels)), wind_speed_colors))
        )

        # 底图：风速填色
        contourf = ax.contourf(
            lon_grid, lat_grid, wind_speed,
            levels=np.linspace(0, 40, 100),
            cmap=cmap,
            transform=ccrs.PlateCarree()
        )

        # 叠加等压线
        contour = ax.contour(
            lon_grid, lat_grid, mslp_smooth,
            levels=np.arange(np.min(mslp_smooth), np.max(mslp_smooth), 1),
            colors='black', linewidths=1.5,
            transform=ccrs.PlateCarree()
        )
        ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f hPa')

        # 叠加风矢（风羽）
        ax.barbs(
            lon_skip, lat_skip, u10_skip, v10_skip,
            length=4, color='black',
            transform=ccrs.PlateCarree()
        )

        # 地理要素与经纬网
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
        gl.top_labels = False
        gl.right_labels = False

        # === 色条：手动创建与图框同高的 Axes，再在其上放置 colorbar ===
        bbox = ax.get_position()  # 当前主图在画布上的位置 [x0, y0, width, height]
        # 在主图右侧新建一个窄轴，高度与主图一致
        cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.02, bbox.height])
        cbar = plt.colorbar(
            contourf, cax=cax, orientation='vertical',
            label='Wind Speed (m/s)',
            ticks=wind_speed_levels
        )

        # 标题（注意：这里的文字是图上的输出，按你原来的英文保留）
        init_dt = datetime.strptime(init_datetime_str, '%Y%m%d%H')

        ax.set_title(
            f'Pangu-{data_source} MSLP (hPa) and 10m Wind (m/s) \n{init_datetime_str} +{forecast_hour:03d}h',
            loc='left', fontsize=14
        )
        ax.set_title(
            f'Max Wind: {max_wind_speed:.2f} m/s\nMin Pressure: {min_pressure:.2f} hPa',
            loc='right', fontsize=14   
        )

        # 保存图片到数据源对应的子文件夹
        filename = f'pangu_mslp_wind_{init_datetime_str}+{forecast_hour:03d}h.png'
        filepath = self._get_output_path(filename, data_source)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

def draw_pangu_results(output_surface_path, init_datetime_str, forecast_hour,
                       data_source='GFS', lon_range=None, lat_range=None):
    """
    顶层接口：读取 Pangu 输出的 NPY 文件，并绘制 MSLP+风场图

    参数:
        output_surface_path: Pangu 地面输出 NPY 文件路径
        init_datetime_str: 起报时间 'YYYYMMDDHH'
        forecast_hour: 预报时效（小时）
        data_source: 数据源名称（用于标题和文件名）
        lon_range: 经度范围 [lon_min, lon_max]
        lat_range: 纬度范围 [lat_min, lat_max]

    返回:
        list[str]: 生成的图片文件路径列表
    """
    drawer = PanguResultDrawer()

    # 读取 NPY 数据
    try:
        raw_data = np.load(output_surface_path)
        data_dict = {
            'mslp': raw_data[0] / 100,  # Pa -> hPa
            'u10': raw_data[1],
            'v10': raw_data[2],
            'raw_data': raw_data
        }
    except Exception as e:
        print(f"[ERROR] 无法读取输出文件: {e}")
        return []

    wind_file = drawer.draw_mslp_and_wind(
        data_dict, init_datetime_str,
        forecast_hour, data_source,
        lon_range, lat_range
    )

    return [wind_file] if wind_file else []


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    # 预报参数
    init_datetime_str = '2025120100'    # 起报时间 YYYYMMDDHH
    forecast_hour = 24                  # 预报时效（小时）
    data_source = 'ERA5'                # 数据源名称

    # 绘图区域参数
    drawing_config = {
        'lon_range': [95, 150],         # 经度范围
        'lat_range': [5, 45],           # 纬度范围
    }
    # =============================================

    # 根据预报参数构造 Pangu 输出文件路径
    # 假设从项目根目录运行，或者 Output 目录在项目根目录
    # 尝试定位项目根目录
    project_root = os.getcwd()
    if not os.path.exists(os.path.join(project_root, 'Output')):
        # 尝试向上查找
        current = script_dir
        for _ in range(4):
            if os.path.exists(os.path.join(current, 'Output')):
                project_root = current
                break
            current = os.path.dirname(current)
            
    filename = f'output_surface_{init_datetime_str}+{forecast_hour:03d}h_{data_source}.npy'
    output_path = os.path.join(project_root, 'Output/Pangu/', filename)
    output_path = os.path.normpath(output_path)

    if os.path.exists(output_path):
        results = draw_pangu_results(
            output_path, init_datetime_str, forecast_hour,
            data_source=data_source,
            lon_range=drawing_config['lon_range'],
            lat_range=drawing_config['lat_range']
        )

        if results:
            for filepath in results:
                print(f"[PLOT] Saved: {filepath}")
    else:
        print(f"[ERROR] Output file not found: {output_path}")