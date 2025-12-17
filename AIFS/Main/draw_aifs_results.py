# AI-weather-models\AIFS\Main\draw_aifs_results.py
# 可视化 AIFS 天气预报结果（生成专业气象图）

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings('ignore')


class AIFSResultDrawer:
    """AIFS 天气预报结果绘图类"""
    
    def __init__(self, output_dir='Run-output-png/AIFS', figsize=(14, 10)):
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
    
    def _get_output_path(self, filename, data_source='ECMWF'):
        """
        根据数据源获取输出路径，自动在数据源子文件夹中保存
        
        参数:
            filename: 图片文件名
            data_source: 数据源名称 (ECMWF, ERA5等)
            
        返回:
            str: 完整的输出文件路径
        """
        source_dir = os.path.join(self.output_dir, data_source.upper())
        os.makedirs(source_dir, exist_ok=True)
        return os.path.join(source_dir, filename)
    
    @staticmethod
    def fix_longitudes(lons):
        """
        经度转换：从 0-360 转换为 -180-180
        
        参数:
            lons: 经度数组
            
        返回:
            np.ndarray: 转换后的经度
        """
        return np.where(lons > 180, lons - 360, lons)
    
    def draw_field_contourf(self, state, field_name, init_datetime_str=None, 
                           lon_range=None, lat_range=None, cmap='RdBu_r', 
                           levels=20, data_source='AIFS'):
        """
        绘制单个气象字段的填色等高线图（使用三角剖分）
        
        参数:
            state: 预报状态字典（包含 date, latitudes, longitudes, fields）
            field_name: 字段名称
            init_datetime_str: 起报时间字符串 'YYYYMMDDHH'
            lon_range: 经度范围 [lon_min, lon_max]，默认全球
            lat_range: 纬度范围 [lat_min, lat_max]，默认全球
            cmap: 颜色映射表
            levels: 等高线级数
            data_source: 数据源标记
            
        返回:
            str: 输出图片文件完整路径，或 None 如果绘制失败
        """
        try:
            if field_name not in state.get('fields', {}):
                print(f"[ERROR] Field '{field_name}' not found in state")
                return None
            
            latitudes = state['latitudes']
            longitudes = state['longitudes']
            values = state['fields'][field_name]
            
            # 确定预报时效
            forecast_hour = int((state['date'] - state.get('init_date', state['date'])).total_seconds() / 3600)
            
            # 构建图像
            fig, ax = plt.subplots(
                figsize=self.figsize,
                subplot_kw={"projection": ccrs.PlateCarree()}
            )
            
            # 地理要素
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
            ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
            
            # 值域处理
            vmin = np.percentile(values[~np.isnan(values)], 2) if not np.all(np.isnan(values)) else 0
            vmax = np.percentile(values[~np.isnan(values)], 98) if not np.all(np.isnan(values)) else 1
            
            # 使用三角剖分绘制数据
            try:
                triangulation = tri.Triangulation(
                    self.fix_longitudes(longitudes),
                    latitudes
                )
                
                contourf = ax.tricontourf(
                    triangulation, values,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    vmin=vmin, vmax=vmax
                )
            except Exception as e:
                print(f"[WARN] Triangulation failed, using regular grid: {e}")
                # 降级方案：直接绘制
                contourf = ax.contourf(
                    self.fix_longitudes(longitudes).reshape(-1),
                    latitudes.reshape(-1),
                    values.reshape(-1),
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree()
                )
            
            # 色条
            cbar = fig.colorbar(contourf, ax=ax, orientation="vertical", shrink=0.7, label=field_name)
            
            # 标题
            if init_datetime_str:
                title = f'AIFS {field_name} at {init_datetime_str} +{forecast_hour:03d}h'
            else:
                title = f'AIFS {field_name} at {state["date"].strftime("%Y-%m-%d %H:%M UTC")} ({forecast_hour:03d}h lead time)'
            
            ax.set_title(title, loc='left', fontsize=12)
            ax.set_title(f'Data source: {data_source}', loc='right', fontsize=10)
            
            # 保存图片到数据源对应的子文件夹
            safe_field_name = field_name.replace('/', '_').replace('\\', '_')
            if init_datetime_str:
                filename = f'aifs_{safe_field_name}_{init_datetime_str}+{forecast_hour:03d}h.png'
            else:
                filename = f'aifs_{safe_field_name}_{state["date"].strftime("%Y%m%d%H")}+{forecast_hour:03d}h.png'
            
            filepath = self._get_output_path(filename, data_source)
            plt.savefig(filepath, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Failed to draw field {field_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def draw_mslp_and_wind(self, state, init_datetime_str=None, 
                          lon_range=None, lat_range=None, data_source='AIFS'):
        """
        绘制海平面气压(MSL)等压线与10m风场（和Pangu一样）
        
        参数:
            state: 预报状态字典
            init_datetime_str: 起报时间字符串
            lon_range: 经度范围
            lat_range: 纬度范围
            data_source: 数据源标记
            
        返回:
            str: 输出文件路径
        """
        try:
            fields = state.get('fields', {})
            
            # 检查必需字段
            if 'msl' not in fields or '100u' not in fields or '100v' not in fields:
                print(f"[ERROR] Required fields not found (msl, 100u, 100v)")
                return None
            
            mslp = fields['msl'] / 100  # Pa -> hPa
            u10 = fields['100u']
            v10 = fields['100v']
            wind_speed = np.sqrt(u10**2 + v10**2)
            
            latitudes = state['latitudes']
            longitudes = state['longitudes']
            
            # 确定预报时效
            forecast_hour = int((state['date'] - state.get('init_date', state['date'])).total_seconds() / 3600)
            
            # 若未指定范围，使用东亚默认范围（同Pangu）
            if lon_range is None or lat_range is None:
                lon_range = [95, 150]
                lat_range = [5, 35]
            
            # 对气压场做高斯平滑
            mslp_smooth = gaussian_filter(mslp, sigma=2)
            
            # 子采样风矢量（每5个格点取一个）
            skip = 5
            u10_skip = u10[::skip, ::skip]
            v10_skip = v10[::skip, ::skip]
            lon_skip = longitudes[::skip, ::skip]
            lat_skip = latitudes[::skip, ::skip]
            
            # 统计量
            max_wind_speed = np.max(wind_speed)
            min_pressure = np.min(mslp)
            
            # 创建图像与地图坐标轴
            fig = plt.figure(figsize=(15, 12))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            # 自定义风速分级及颜色（和Pangu一样）
            wind_speed_color_map = {
                0: '#FFFFFFFF', 5: '#D1D1D1FF', 10: '#188FD8FF', 15: '#34D259FF',
                20: '#F1B04DFF', 25: '#FF6200FF', 30: '#FF0000FF', 35: '#9C2EBAFF',
                40: '#FF00FFFF', 45: '#EA00FFFF'
            }
            wind_speed_levels = list(wind_speed_color_map.keys())
            wind_speed_colors = list(wind_speed_color_map.values())
            
            cmap = LinearSegmentedColormap.from_list(
                'wind_speed_cmap',
                list(zip(np.linspace(0, 1, len(wind_speed_levels)), wind_speed_colors))
            )
            
            # 底图：风速填色
            contourf = ax.contourf(
                self.fix_longitudes(longitudes), latitudes, wind_speed,
                levels=np.linspace(0, 40, 100),
                cmap=cmap,
                transform=ccrs.PlateCarree()
            )
            
            # 叠加等压线
            contour = ax.contour(
                self.fix_longitudes(longitudes), latitudes, mslp_smooth,
                levels=np.arange(np.min(mslp_smooth), np.max(mslp_smooth), 1),
                colors='black', linewidths=1.5,
                transform=ccrs.PlateCarree()
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f hPa')
            
            # 叠加风矢（风羽）
            ax.barbs(
                self.fix_longitudes(lon_skip), lat_skip, u10_skip, v10_skip,
                length=4, color='black',
                transform=ccrs.PlateCarree()
            )
            
            # 地理要素与经纬网
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False
            
            # === 色条：手动创建与图框同高的 Axes ===
            bbox = ax.get_position()
            cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.02, bbox.height])
            cbar = plt.colorbar(
                contourf, cax=cax, orientation='vertical',
                label='Wind Speed (m/s)',
                ticks=wind_speed_levels
            )
            
            # 标题
            ax.set_title(
                f'AIFS MSLP (hPa) and 10m Wind (m/s) \n{init_datetime_str} +{forecast_hour:03d}h',
                loc='left', fontsize=14
            )
            ax.set_title(
                f'Max Wind: {max_wind_speed:.2f} m/s\nMin Pressure: {min_pressure:.2f} hPa',
                loc='right', fontsize=14   
            )
            
            # 保存图片
            filename = f'aifs_mslp_wind_{init_datetime_str}+{forecast_hour:03d}h_{data_source}.png'
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"[OK] Saved: {filename}")
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Failed to draw MSLP and wind: {e}")
            import traceback
            traceback.print_exc()
            return None


def draw_aifs_results(state, init_datetime_str=None, data_source='AIFS',
                     fields_to_draw=None, draw_wind=True):
    """
    顶层接口：读取 AIFS 预报状态，并绘制MSLP+10m风场图（和Pangu一样）
    
    参数:
        state: AIFS 预报状态字典
        init_datetime_str: 起报时间 'YYYYMMDDHH'
        data_source: 数据源名称
        fields_to_draw: 保留以兼容（不使用）
        draw_wind: 保留以兼容（不使用）
        
    返回:
        list[str]: 生成的图片文件路径列表
    """
    drawer = AIFSResultDrawer()
    
    image_files = []
    
    # 绘制MSLP + 10m风场（和Pangu一样）
    filepath = drawer.draw_mslp_and_wind(
        state,
        init_datetime_str=init_datetime_str,
        data_source=data_source
    )
    if filepath:
        image_files.append(filepath)
    
    return image_files


if __name__ == '__main__':
    # 示例用法
    from run_aifs import AIFSRunner
    from get_data_aifs import get_aifs_data
    
    # 获取初始条件
    print("[INFO] Retrieving initial conditions...")
    input_state = get_aifs_data()
    
    if input_state is None:
        print("[ERROR] Failed to retrieve initial conditions")
        sys.exit(1)
    
    # 运行预报
    print("[INFO] Running forecast...")
    runner = AIFSRunner(device='cuda')
    
    # 绘制结果
    print("[INFO] Drawing results...")
    for state in runner.runner.run(input_state=input_state, lead_time=3):
        image_files = draw_aifs_results(
            state,
            init_datetime_str=input_state['date'].strftime('%Y%m%d%H')
        )
        
        print(f"\nGenerated {len(image_files)} images")
        for img_file in image_files:
            print(f"  - {img_file}")
