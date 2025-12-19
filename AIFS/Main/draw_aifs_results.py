# AI-weather-models\AIFS\Main\draw_aifs_results.py
# 可视化 AIFS 天气预报结果（生成专业气象图）
# 参考 Pangu 的绘图方式，处理 AIFS 的 N320 非结构化网格数据

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from scipy.ndimage import gaussian_filter
import warnings

warnings.filterwarnings("ignore")

try:
    import earthkit.regrid as ekr
except ImportError:
    print("[ERROR] earthkit-regrid not installed, run: pip install earthkit-regrid")
    sys.exit(1)


class AIFSResultDrawer:
    """AIFS 天气预报结果绘图类 - 使用 ekr.interpolate 插值方法（参考 main.py）"""

    def __init__(self, output_dir="Run-output-png/AIFS", figsize=(15, 12)):
        # 自动定位项目根目录
        if not os.path.isabs(output_dir):
            project_root = os.getcwd()
            if not os.path.exists(os.path.join(project_root, "Run-output-png")):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                current = script_dir
                for _ in range(4):
                    if os.path.exists(os.path.join(current, "Run-output-png")):
                        project_root = current
                        break
                    current = os.path.dirname(current)
            output_dir = os.path.normpath(os.path.join(project_root, output_dir))

        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_output_path(self, filename, data_source="ECMWF"):
        source_dir = os.path.join(self.output_dir, data_source.upper())
        os.makedirs(source_dir, exist_ok=True)
        return os.path.join(source_dir, filename)

    def draw_mslp_and_wind(self, state, init_datetime_str=None, data_source="AIFS", 
                          lon_range=None, lat_range=None):
        """
        绘制 MSLP + 10m 风场图（参考 Pangu 的绘图方式）
        
        参数:
            state: AIFS 模型状态字典
            init_datetime_str: 初始化时间字符串
            data_source: 数据源标识
            lon_range: 绘图经度范围 [lon_min, lon_max]，None 表示全球
            lat_range: 绘图纬度范围 [lat_min, lat_max]，None 表示全球
        """
        try:
            # 基础校验
            fields = state.get("fields", {})
            if "msl" not in fields or "10u" not in fields or "10v" not in fields:
                print("[ERROR] Required fields not found (msl, 10u, 10v)")
                return None

            # 读取并处理数据（处理 N320 非结构化网格）
            def _first_and_flat(arr, arr_name):
                """展平数组为1D（处理可能的时间维度）"""
                if not isinstance(arr, np.ndarray):
                    raise ValueError(f"{arr_name} is not ndarray")
                if arr.ndim > 1:
                    if arr.shape[0] > 1:
                        arr = arr[0]
                    arr = arr.reshape(-1)
                else:
                    arr = arr.reshape(-1)
                return arr

            mslp_n320 = _first_and_flat(fields["msl"], "msl") / 100.0  # Pa -> hPa
            u10_n320 = _first_and_flat(fields["10u"], "10u")
            v10_n320 = _first_and_flat(fields["10v"], "10v")

            print(f"[INFO] Processing N320 data with {mslp_n320.size} grid points")

            # N320 → 0.25x0.25 度插值
            print(f"[INFO] Interpolating N320 -> (0.25, 0.25) grid...")
            mslp_grid = ekr.interpolate(mslp_n320, {"grid": "N320"}, {"grid": (0.25, 0.25)})
            u10_grid = ekr.interpolate(u10_n320, {"grid": "N320"}, {"grid": (0.25, 0.25)})
            v10_grid = ekr.interpolate(v10_n320, {"grid": "N320"}, {"grid": (0.25, 0.25)})
            
            # 转换为 numpy 数组并确保二维
            mslp_grid = np.asarray(mslp_grid).squeeze()
            u10_grid = np.asarray(u10_grid).squeeze()
            v10_grid = np.asarray(v10_grid).squeeze()
            
            print(f"[DEBUG] Interpolated shapes: mslp={mslp_grid.shape}, u10={u10_grid.shape}, v10={v10_grid.shape}")
            
            # 诊断 NaN
            print(f"\n[DIAGNOSTIC] Checking NaN in interpolated data:")
            for name, data in [("MSLP", mslp_grid), ("U10", u10_grid), ("V10", v10_grid)]:
                nan_count = np.isnan(data).sum()
                nan_pct = 100 * nan_count / data.size if data.size > 0 else 0
                print(f"  {name}: NaN count={nan_count} ({nan_pct:.2f}%)")
            
            # 构建网格（标准 0.25x0.25 度全球网格）
            lat = np.linspace(90, -90, mslp_grid.shape[0])
            lon = np.linspace(0, 360, mslp_grid.shape[1])
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            
            # 计算风速
            wind_speed = np.sqrt(u10_grid**2 + v10_grid**2)
            
            # 应用高斯平滑（参考 Pangu）
            print("[INFO] Applying Gaussian smoothing...")
            mslp_smooth = gaussian_filter(mslp_grid, sigma=2.0)
            wind_speed_smooth = gaussian_filter(wind_speed, sigma=2.0)
            
            # 如果指定了范围，则提取该范围内的数据；否则使用全球
            if lon_range is not None and lat_range is not None:
                lon_min, lon_max = lon_range
                lat_min, lat_max = lat_range
                lon_idx = np.where((lon_grid[0, :] >= lon_min) & (lon_grid[0, :] <= lon_max))[0]
                lat_idx = np.where((lat_grid[:, 0] >= lat_min) & (lat_grid[:, 0] <= lat_max))[0]
                
                mslp_smooth = mslp_smooth[np.ix_(lat_idx, lon_idx)]
                wind_speed_smooth = wind_speed_smooth[np.ix_(lat_idx, lon_idx)]
                u10_grid = u10_grid[np.ix_(lat_idx, lon_idx)]
                v10_grid = v10_grid[np.ix_(lat_idx, lon_idx)]
                lon_grid = lon_grid[np.ix_(lat_idx, lon_idx)]
                lat_grid = lat_grid[np.ix_(lat_idx, lon_idx)]
                print(f"[INFO] Extracted region: lon=[{lon_min}, {lon_max}], lat=[{lat_min}, {lat_max}]")
            
            max_wind_speed = np.max(wind_speed_smooth)
            min_pressure = np.min(mslp_smooth)

            # 计算预报时效
            forecast_hour = int(
                (state["date"] - state.get("init_date", state["date"])).total_seconds() / 3600
            )

            # 绘图（参考 Pangu 的绘图方式）
            fig = plt.figure(figsize=self.figsize)
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)

            # 风速颜色映射
            wind_speed_color_map = {
                0: "#FFFFFFFF",
                5: "#D1D1D1FF",
                10: "#188FD8FF",
                15: "#34D259FF",
                20: "#F1B04DFF",
                25: "#FF6200FF",
                30: "#FF0000FF",
                35: "#9C2EBAFF",
                40: "#FF00FFFF",
                45: "#EA00FFFF",
            }
            wind_speed_levels = list(wind_speed_color_map.keys())
            wind_speed_colors = list(wind_speed_color_map.values())
            cmap = LinearSegmentedColormap.from_list(
                "wind_speed_cmap",
                list(zip(np.linspace(0, 1, len(wind_speed_levels)), wind_speed_colors)),
            )

            # 风速填色
            contourf = ax.contourf(
                lon_grid,
                lat_grid,
                wind_speed_smooth,
                levels=np.linspace(0, 40, 100),
                cmap=cmap,
                transform=ccrs.PlateCarree(),
            )

            # 等压线
            contour = ax.contour(
                lon_grid,
                lat_grid,
                mslp_smooth,
                levels=np.arange(np.min(mslp_smooth), np.max(mslp_smooth), 1),
                colors="black",
                linewidths=1.5,
                transform=ccrs.PlateCarree(),
            )
            ax.clabel(contour, inline=True, fontsize=8, fmt="%1.0f hPa")

            # 经纬网
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.3)
            gl.top_labels = False
            gl.right_labels = False

            # 色条
            bbox = ax.get_position()
            cax = fig.add_axes([bbox.x1 + 0.015, bbox.y0, 0.02, bbox.height])
            plt.colorbar(
                contourf,
                cax=cax,
                orientation="vertical",
                label="Wind Speed (m/s)",
                ticks=wind_speed_levels,
            )

            # 标题
            ax.set_title(
                f"AIFS MSLP (hPa) and 10m Wind (m/s)\n{init_datetime_str} +{forecast_hour:03d}h",
                loc="left",
                fontsize=14,
            )
            ax.set_title(
                f"Max Wind: {max_wind_speed:.2f} m/s\nMin Pressure: {min_pressure:.2f} hPa",
                loc="right",
                fontsize=14,
            )

            # 保存
            filename = f"aifs_mslp_wind_{init_datetime_str}+{forecast_hour:03d}h_{data_source}.png"
            filepath = self._get_output_path(filename, data_source)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()
            
            print(f"[OK] Saved: {filename}")
            return filepath

        except Exception as e:
            print(f"[ERROR] Failed to draw MSLP and wind: {e}")
            import traceback
            traceback.print_exc()
            return None


def draw_aifs_results(state, init_datetime_str=None, data_source="AIFS",
                      use_gpu_interp=False, target_res=0.5, device='cuda'):
    """
    顶层接口：绘制 AIFS 预报结果
    
    参数:
        state: AIFS 模型状态字典
        init_datetime_str: 初始时间字符串
        data_source: 数据源标识
        use_gpu_interp: 保留参数（兼容性），当前未使用
        target_res: 保留参数（兼容性），当前未使用
        device: 保留参数（兼容性），当前未使用
    
    返回:
        image_files: 生成的图片文件路径列表
    """
    drawer = AIFSResultDrawer()
    image_files = []
    
    filepath = drawer.draw_mslp_and_wind(
        state, 
        init_datetime_str=init_datetime_str, 
        data_source=data_source
    )
    
    if filepath:
        image_files.append(filepath)
    
    return image_files


if __name__ == "__main__":
    from run_aifs import AIFSRunner
    from get_data_aifs import get_aifs_data

    print("[INFO] Retrieving initial conditions...")
    input_state = get_aifs_data()
    if input_state is None:
        print("[ERROR] Failed to retrieve initial conditions")
        sys.exit(1)

    print("[INFO] Running forecast...")
    runner = AIFSRunner(device="cuda")

    print("[INFO] Drawing results...")
    for state in runner.runner.run(input_state=input_state, lead_time=12):
        state['init_date'] = input_state['date']
        image_files = draw_aifs_results(
            state, 
            init_datetime_str=input_state["date"].strftime("%Y%m%d%H")
        )
        print(f"Generated {len(image_files)} images")
        for img_file in image_files:
            print(f"  - {img_file}")
