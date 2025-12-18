# AI-weather-models\AIFS\Main\draw_aifs_results.py
# 可视化 AIFS 天气预报结果（生成专业气象图）
# 参考 main.py 的方法，使用 ekr.interpolate 插值而非三角剖分（避免卡死）

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

try:
    import earthkit.regrid as ekr
except ImportError:
    print("[ERROR] earthkit-regrid not installed, run: pip install earthkit-regrid")
    sys.exit(1)


def _fix_longitudes(lons):
    """将经度 0-360 转换为 -180-180（与notebook一致）"""
    return np.where(lons > 180, lons - 360, lons)


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

    def draw_mslp_and_wind(self, state, init_datetime_str=None, data_source="AIFS"):
        """
        绘制 MSLP + 10m 风场图
        参考 main.py 使用 ekr.interpolate 插值到规则网格（避免三角剖分卡死）
        """
        try:
            # 基础校验
            fields = state.get("fields", {})
            if "msl" not in fields or "10u" not in fields or "10v" not in fields:
                print("[ERROR] Required fields not found (msl, 10u, 10v)")
                print(f"[DEBUG] Available fields: {list(fields.keys())[:10]}...")
                return None

            # 读取数据（N320 非结构化网格数据）
            def _first_and_flat(arr, arr_name):
                """展平数组为1D（处理可能的时间维度）"""
                if not isinstance(arr, np.ndarray):
                    raise ValueError(f"{arr_name} is not ndarray")
                if arr.ndim > 1:
                    if arr.shape[0] > 1:
                        arr = arr[0]  # 取第一个时间步
                    arr = arr.reshape(-1)
                else:
                    arr = arr.reshape(-1)
                return arr

            mslp_n320 = _first_and_flat(fields["msl"], "msl") / 100.0  # Pa -> hPa
            u10_n320 = _first_and_flat(fields["10u"], "10u")
            v10_n320 = _first_and_flat(fields["10v"], "10v")

            print(f"[INFO] Processing N320 data with {mslp_n320.size} grid points")

            # 使用 ekr.interpolate 将 N320 非结构化网格插值到规则网格（参考 main.py）
            # main.py: ekr.interpolate(ds['fields']['2t'], {"grid": "N320"}, {"grid": (0.25, 0.25)})
            target_res = (0.25, 0.25)  # 目标分辨率：0.25度
            print(f"[INFO] Interpolating N320 -> {target_res} grid...")
            
            mslp_grid = ekr.interpolate(mslp_n320, {"grid": "N320"}, {"grid": target_res})
            u10_grid = ekr.interpolate(u10_n320, {"grid": "N320"}, {"grid": target_res})
            v10_grid = ekr.interpolate(v10_n320, {"grid": "N320"}, {"grid": target_res})
            
            # 生成规则网格的经纬度（参考 main.py）
            # main.py: latitudes = np.arange(90,-90.25, -0.25), longitudes = fix(np.arange(0, 360, 0.25))
            latitudes = np.arange(90, -90.25, -target_res[1])
            longitudes = np.arange(0, 360, target_res[0])
            lons, lats = np.meshgrid(longitudes, latitudes)
            
            # 修正经度为 -180 到 180（参考 main.py 的 fix 函数）
            lons_fixed = _fix_longitudes(lons)
            
            print(f"[INFO] Interpolation completed: grid shape = {mslp_grid.shape}")

            # 计算风速
            wind_speed = np.sqrt(u10_grid**2 + v10_grid**2)
            max_wind_speed = float(np.max(wind_speed))
            min_pressure = float(np.min(mslp_grid))

            # 计算预报时效
            forecast_hour = int(
                (state["date"] - state.get("init_date", state["date"])).total_seconds() / 3600
            )

            # 绘图（使用规则网格，不需要三角剖分）
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

            # 风速填色（使用规则网格的 contourf）
            contourf = ax.contourf(
                lons_fixed,
                lats,
                wind_speed,
                levels=np.linspace(0, 40, 100),
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend="both",
            )

            # 等压线
            contour = ax.contour(
                lons_fixed,
                lats,
                mslp_grid,
                levels=np.arange(960, 1040, 4),
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
