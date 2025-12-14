import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable

def draw_path(output_surface_files, datetime, data_source, model_type, i, output_path_png_name, lon_min, lon_max, lat_min, lat_max):
    lons, lats, pressures = [], [], []
    # 设置字体
    import matplotlib.font_manager as font_manager
    font_manager.fontManager.addfont(f'AImodels\MiSans VF.ttf')
    plt.rcParams['font.sans-serif'] = ['MiSans VF']
    
    for file in output_surface_files:
        data = np.load(file)
        # 提取气压数据
        mslp = data[0] / 100  # 转换为hPa
        # 创建经纬度网格
        lon = np.linspace(0, 360, mslp.shape[1])
        lat = np.linspace(90, -90, mslp.shape[0])
        lon, lat = np.meshgrid(lon, lat)

        # 提取指定范围的索引
        lon_indices = np.where((lon[0, :] >= lon_min) & (lon[0, :] <= lon_max))[0]
        lat_indices = np.where((lat[:, 0] >= lat_min) & (lat[:, 0] <= lat_max))[0]
        
        # 提取指定范围内的气压数据
        mslp_subset = mslp[np.ix_(lat_indices, lon_indices)]
        lon_subset = lon[np.ix_(lat_indices, lon_indices)]
        lat_subset = lat[np.ix_(lat_indices, lon_indices)]
        
        # 找到最低气压的索引
        min_pressure_idx = np.unravel_index(np.argmin(mslp_subset), mslp_subset.shape)
        lons.append(lon_subset[min_pressure_idx])
        lats.append(lat_subset[min_pressure_idx])
        pressures.append(mslp_subset[min_pressure_idx])
    
    # 找到路径点中的最低气压
    min_pressure = min(pressures)
    
    plt.figure(figsize=(15, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='#FFFFFFFF')
    ax.add_feature(cfeature.OCEAN, facecolor='#A8E0FFFF')
    ax.coastlines()
    
    # 设置网格线并禁用顶部的经度标签
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False

    # 设定气压的等级和对应的颜色
    pressure_color_map = {
        1030: '#FFFFFFFF',
        1010: '#D1D1D1FF',
        1000: '#188FD8FF',
        990: '#51FF00FF',
        980: '#F1B04DFF',
        970: '#FF6200FF',
        950: '#FF0000FF',
        930: '#9C2EBAFF',
        910: '#FF00FFFF',
        890: '#EA00FFFF'
    }
    pressure_levels = list(pressure_color_map.keys())
    pressure_colors = list(pressure_color_map.values())

    # 创建连续的颜色映射
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('pressure_cmap', list(zip(np.linspace(0, 1, len(pressure_levels)), pressure_colors[::-1])))
    # 绘制台风路径
    plt.plot(lons, lats, 'k--', transform=ccrs.PlateCarree())
    sc = ax.scatter(lons, lats, c=pressures, cmap=cmap, s=100, transform=ccrs.PlateCarree(), vmin=min(pressure_levels), vmax=max(pressure_levels))
    # 设置标题，标出路径点中的最低气压和当前推理时间
    hour = model_type*i
    plt.title(f'Pangu Typhoon Path Based on {data_source} {datetime} +{hour:02d}h', fontsize=16, loc='left')
    plt.title(f'Lowest Pressure: {min_pressure:.2f} hPa ', fontsize=16, loc='right')
    
    # 创建色条并使其与图像高度一致
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1, axes_class=plt.Axes)
    cbar = plt.colorbar(sc, cax=cax)
    cbar.set_label('Pressure (hPa)')

    # 调整四周边距
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    plt.savefig(output_path_png_name)
    plt.close()