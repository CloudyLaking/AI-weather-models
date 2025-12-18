# AI-weather-models\AIFS\Main\draw_aifs_results.py
# 可视化 AIFS 天气预报结果（生成专业气象图）
# 参考 run_AIFS_v1.ipynb 的原始做法

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")


def _fix_longitudes(lons):
    """将经度 0-360 转换为 -180-180（与notebook一致）"""
    return np.where(lons > 180, lons - 360, lons)


class AIFSResultDrawer:
    """AIFS 天气预报结果绘图类 - 使用notebook原始方法"""

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
        使用与 notebook 完全一致的方法：tri.Triangulation
        """
        try:
            # 基础校验
            if "latitudes" not in state or "longitudes" not in state:
                print("[ERROR] State missing latitudes/longitudes")
                return None
            
            fields = state.get("fields", {})
            if "msl" not in fields or "10u" not in fields or "10v" not in fields:
                print("[ERROR] Required fields not found (msl, 10u, 10v)")
                print(f"[DEBUG] Available fields: {list(fields.keys())[:10]}...")
                return None

            # 读取经纬度与数据（与notebook一致的处理方式）
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

            latitudes = _first_and_flat(state["latitudes"], "latitudes")
            longitudes = _first_and_flat(state["longitudes"], "longitudes")
            
            mslp = _first_and_flat(fields["msl"], "msl") / 100.0  # Pa -> hPa
            u10 = _first_and_flat(fields["10u"], "10u")
            v10 = _first_and_flat(fields["10v"], "10v")

            # 数据验证
            if not (latitudes.size == longitudes.size == mslp.size == u10.size == v10.size):
                print(
                    "[ERROR] Mismatched sizes: "
                    f"lat={latitudes.size}, lon={longitudes.size}, "
                    f"msl={mslp.size}, u10={u10.size}, v10={v10.size}"
                )
                return None

            print(f"[INFO] Processing {latitudes.size} grid points")

            # 计算风速
            wind_speed = np.sqrt(u10**2 + v10**2)
            max_wind_speed = float(np.max(wind_speed))
            min_pressure = float(np.min(mslp))

            # 计算预报时效
            forecast_hour = int(
                (state["date"] - state.get("init_date", state["date"])).total_seconds() / 3600
            )

            # 创建三角剖分（与notebook完全一致）
            print(f"[INFO] Creating Delaunay triangulation...")
            import time
            t0 = time.time()
            
            lons_fixed = _fix_longitudes(longitudes)
            triangulation = tri.Triangulation(lons_fixed, latitudes)
            
            print(f"[INFO] Triangulation completed in {time.time()-t0:.2f}s")

            # 绘图
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

            # 风速填色（使用三角网格）
            contourf = ax.tricontourf(
                triangulation,
                wind_speed,
                levels=np.linspace(0, 40, 100),
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend="both",
            )

            # 等压线
            contour = ax.tricontour(
                triangulation,
                mslp,
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
