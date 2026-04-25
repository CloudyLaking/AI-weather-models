import warnings
# Ignore UserWarning from pyproj and FutureWarning from cfgrib/xarray
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import urllib.request
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
import numpy as np
from datetime import datetime

def plot_aifs_snow(lon_range=(100, 125), lat_range=(22, 42),
                   init_time="20250214120000", lead_time="216h", download=True):
    """
    Draw AIFS snowfall data plot.
    
    Parameters:
      lon_range: Longitude range (default: [110, 130])
      lat_range: Latitude range (default: [25, 37])
      init_time: Initialization time as a string, for example "20250214120000"
      lead_time: Lead time as a string, for example "216h"
      download: Whether to download the file (default: True)
    """
    # Set font path and load font
    font_path = r"AI-weather-models/MiSans VF.ttf"
    myfont = fm.FontProperties(fname=font_path)

    # Construct download URL and local save path — extract date from init_time
    filename = f"{init_time}-{lead_time}-oper-fc.grib2"
    url = f"https://data.ecmwf.int/forecasts/{init_time[:8]}/{init_time[8:10]}z/aifs-single/0p25/oper/{filename}"
    local_dir = r"aifs-open-data"
    local_file = os.path.join(local_dir, filename)

    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)

    if download:
        if os.path.exists(local_file):
            print(f"✓ 数据文件已存在: {local_file}")
        else:
            print(f"📥 正在下载文件: {url}")
            try:
                urllib.request.urlretrieve(url, local_file)
                print(f"✓ 文件已下载至: {local_file}")
            except Exception as e:
                print(f"✗ 下载失败: {e}")
                return
    else:
        # 如果不下载，检查文件是否存在
        if not os.path.exists(local_file):
            print(f"✗ 文件不存在且download=False: {local_file}")
            return
        print(f"✓ 使用已存在的文件: {local_file}")

    # Read data using cfgrib engine, select variable with shortName 'sf' to avoid merge conflicts
    ds = xr.open_dataset(
        local_file,
        engine="cfgrib",
        backend_kwargs={'filter_by_keys': {'shortName': 'sf'}}
    )

    # Select data based on latitude and longitude range
    subset = ds.where(
        (ds.latitude >= lat_range[0]) & (ds.latitude <= lat_range[1]) &
        (ds.longitude >= lon_range[0]) & (ds.longitude <= lon_range[1]),
        drop=True
    )

    # Extract snowfall data (variable 'sf')
    sf_data = subset['sf']
    
    # 获取最大值
    max_snowfall = float(sf_data.max().values)

    # Create Cartopy map with PlateCarree projection
    plt.figure(figsize=(11, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]], crs=ccrs.PlateCarree())

    # Add background features: land, ocean, coastline, and rivers
    ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN.with_scale('50m'))
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='blue')

    # 添加中国的省份边界（使用 Natural Earth 中的 admin_1_states_provinces_lines 数据）
    provinces = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lines',
        scale='50m',
        facecolor='none')
    ax.add_feature(provinces, edgecolor='black', linewidth=0.6, linestyle='--')

    # Add gridlines and labels
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 设置更丰富的颜色映射和标准化
    levels = [0, 0.1, 0.5, 1, 2, 5, 10, 15, 25, 40, 60, 100]  # 毫米单位
    colors = [
        '#FFFFFF',   # 0-0.1 mm: 白色
        '#E8F4FF',   # 0.1-0.5 mm: 极浅蓝
        '#D0EDFF',   # 0.5-1 mm: 浅蓝
        '#A8DBFF',   # 1-2 mm: 浅蓝
        '#6FC3FF',   # 2-5 mm: 中浅蓝
        '#0096FF',   # 5-10 mm: 亮蓝
        '#0070FF',   # 10-15 mm: 深蓝
        '#4000FF',   # 15-25 mm: 蓝紫
        "#5700C2",   # 25-40 mm: 洋红
        "#9A003D",   # 40-60 mm: 深洋红
        "#CF00AD"    # 60+ mm: 深红
    ]
    custom_cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, custom_cmap.N)

    # 使用 meshgrid 将一维经纬度转换为二维数组
    lon2d, lat2d = np.meshgrid(subset.longitude, subset.latitude)
    cf = ax.contourf(lon2d, lat2d, sf_data, levels=levels,
                     cmap=custom_cmap, norm=norm, transform=ccrs.PlateCarree())

    # 创建带上尖的 colorbar
    cbar_ax = plt.axes([0.92, 0.15, 0.04, 0.7])
    cb = plt.colorbar(cf, cax=cbar_ax, orientation="vertical", pad=0.02)
    cb.set_label("Snowfall (mm)", fontproperties=myfont, fontsize=11)
    
    # Set main title (left aligned) and axis labels (using the custom font)
    ax.text(0.01, 1.05, "AIFS Total Snowfall Equivalent Water Content (mm)", transform=ax.transAxes,
            fontproperties=myfont, fontsize=16, ha='left', va='bottom', weight='bold')
    ax.set_xlabel("Longitude", fontproperties=myfont)
    ax.set_ylabel("Latitude", fontproperties=myfont)

    # 解析时间信息
    init_dt = datetime.strptime(init_time, "%Y%m%d%H%M%S")
    init_str = init_dt.strftime("%Y-%m-%d %HZ")
    lead_hours = int(lead_time.replace('h', ''))
    
    # Add subtitle showing init time and lead time
    ax.text(0.015, 1.01, f"Init: {init_str} | Forecast Hour: {lead_hours}h",
            transform=ax.transAxes, fontproperties=myfont, fontsize=11,
            ha='left', va='bottom')
    
    # 添加最大值信息 (format same as By CloudyLake)
    ax.text(0.995, 1.055, f"Maximum: {max_snowfall:.2f} mm",
            transform=ax.transAxes, fontproperties=myfont, fontsize=12,
            ha='right', va='bottom', weight='bold')
    
    # 作者信息
    ax.text(0.995, 1.01, f"By CloudyLake",
            transform=ax.transAxes, fontproperties=myfont, fontsize=12,
            ha='right', va='bottom', weight='bold')
    
    # 保存图片至输出目录
    output_dir = r"aifs-open-data-output-png"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"aifs_snow_{init_time}_{lead_time}.png")
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"✓ 图片已保存至: {output_file}")
    plt.close()

if __name__ == "__main__":
    # Default longitude and latitude range: [110, 130, 25, 37]. Modify init_time and lead_time as needed.
    plot_aifs_snow(lon_range=(100, 135), lat_range=(23, 45),
                   init_time="20260424000000", lead_time="360h", download=True)