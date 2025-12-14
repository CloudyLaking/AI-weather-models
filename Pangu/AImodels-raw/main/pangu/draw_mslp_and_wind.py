import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from matplotlib.font_manager import FontProperties
import os

def draw_mslp_and_wind(datetime ='2021021700', model_type =  6, run_times=3, file_path='output_data\pangu', output_image_path='EC_archive\script', hour=0, lon_min=95, lon_max=150, lat_min=5, lat_max=35, data_source='ERA5'):    
    # 加载 .npy 文件
    data = np.load(file_path)

    # 提取气压和风场数据
    mslp = data[0] / 100  # 转换为hPa
    u10 = data[1]
    v10 = data[2]

    # 计算风速
    wind_speed = np.sqrt(u10**2 + v10**2)

    # 创建经纬度网格
    lon = np.linspace(0, 360, mslp.shape[1])
    lat = np.linspace(90, -90, mslp.shape[0])
    lon, lat = np.meshgrid(lon, lat)

    # 提取指定范围的索引
    lon_indices = np.where((lon[0, :] >= lon_min) & (lon[0, :] <= lon_max))[0]
    lat_indices = np.where((lat[:, 0] >= lat_min) & (lat[:, 0] <= lat_max))[0]

    # 提取指定范围的数据
    region_mslp = mslp[np.ix_(lat_indices, lon_indices)]
    region_u10 = u10[np.ix_(lat_indices, lon_indices)]
    region_v10 = v10[np.ix_(lat_indices, lon_indices)]
    region_wind_speed = wind_speed[np.ix_(lat_indices, lon_indices)]
    region_lon = lon[np.ix_(lat_indices, lon_indices)]
    region_lat = lat[np.ix_(lat_indices, lon_indices)]

    # 对气压数据进行平滑处理
    region_mslp_smooth = gaussian_filter(region_mslp, sigma=2)

    # 下采样以减少风旗密度
    skip = (slice(None, None, 5), slice(None, None, 5))  # 每隔5个点取一个
    region_u10_skip = region_u10[skip]
    region_v10_skip = region_v10[skip]
    region_lon_skip = region_lon[skip]
    region_lat_skip = region_lat[skip]

    # 计算最大风速和最低气压
    max_wind_speed = np.max(region_wind_speed)
    min_pressure = np.min(region_mslp)
    print(f"Max Wind Speed: {max_wind_speed:.2f} m/s")
    print(f"Min Pressure: {min_pressure:.2f} hPa")
    
    # 设置字体
    import matplotlib.font_manager as font_manager
    font_manager.fontManager.addfont(f'AImodels\MiSans VF.ttf')
    plt.rcParams['font.sans-serif'] = ['MiSans VF']

    # 创建绘图
    fig = plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设定风速的等级和对应的颜色
    wind_speed_color_map = {
        0: '#FFFFFFFF',
        5: '#D1D1D1FF',
        10: '#188FD8FF',
        15: '#34D259FF',
        20: '#F1B04DFF',
        25: '#FF6200FF',
        30: '#FF0000FF',
        35: '#9C2EBAFF',
        40: '#FF00FFFF',
        45: '#EA00FFFF'
    }
    wind_speed_levels = list(wind_speed_color_map.keys())
    wind_speed_colors = list(wind_speed_color_map.values())

    # 创建连续的颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('wind_speed_cmap', list(zip(np.linspace(0, 1, len(wind_speed_levels)), wind_speed_colors)))

    # 依据风速大小上色
    contourf = ax.contourf(region_lon, region_lat, region_wind_speed, levels=np.linspace(0, 40, 100), cmap=cmap, transform=ccrs.PlateCarree())
    # 添加气压等值线
    contour = ax.contour(region_lon, region_lat, region_mslp_smooth, levels=np.arange(np.min(region_mslp_smooth), np.max(region_mslp_smooth), 1), colors='black', linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f hPa')

    # 绘制风旗
    ax.barbs(region_lon_skip, region_lat_skip, region_u10_skip, region_v10_skip, length=4, color='black')

    # 添加海岸线和边界
    ax.coastlines()

    # 添加经纬度网格
    gl = ax.gridlines(draw_labels=True, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # 创建颜色条轴并添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)

    # 设置色条数据点间距
    cbar = plt.colorbar(contourf, cax=cax, orientation='vertical', label='Wind Speed (m/s)', ticks=wind_speed_levels)
    # 添加标题
    ax.set_title(f'Pangu-{data_source} MSLP (hPa) and 10m Wind (m/s) {datetime} +{hour}h ', loc='left', fontsize=16)
    ax.set_title(f'Max Wind Speed: {max_wind_speed:.2f} m/s  Min Pressure: {min_pressure:.2f} hPa', loc='right', fontsize=16)

    # 调整布局以减少白色留白
    plt.tight_layout()

    # 确保输出目录存在
    output_dir = os.path.dirname(output_image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 保存图像
    plt.savefig(output_image_path, bbox_inches='tight', dpi=300)
    plt.close()

#draw_mslp_and_wind(datetime ='2021021700', model_type =  6, run_times=3, file_path=r'Pangu_forecast\temp_output\output_surface_24h.npy', output_image_path='EC_archive\script', hour=0, lon_min=120, lon_max=150, lat_min=5, lat_max=20, data_source='ERA5')