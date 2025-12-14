import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter

# 指定 .npy 文件的路径
file_path = r'main\input\input_surface.npy'

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

# 指定经纬度范围
lon_min, lon_max = 110, 160
lat_min, lat_max = 0, 30

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

# 对气压数据进行平滑处理，增加 sigma 值
region_mslp_smooth = gaussian_filter(region_mslp, sigma=3)

# 下采样以减少风矢密度
skip = (slice(None, None, 5), slice(None, None, 5))  # 每隔5个点取一个
region_u10_skip = region_u10[skip]
region_v10_skip = region_v10[skip]
region_lon_skip = region_lon[skip]
region_lat_skip = region_lat[skip]

# 计算最大风速和最低气压
max_wind_speed = np.max(region_wind_speed)
min_pressure = np.min(region_mslp)

# 创建绘图
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# 绘制风速等值线并依据风速大小上色
contourf = ax.contourf(region_lon, region_lat, region_wind_speed, cmap='coolwarm', transform=ccrs.PlateCarree())
contour = ax.contour(region_lon, region_lat, region_mslp_smooth, levels=np.arange(np.min(region_mslp_smooth), np.max(region_mslp_smooth), 2), colors='black')
ax.clabel(contour, inline=True, fontsize=8, fmt='%1.0f hPa')

# 绘制风场，箭头颜色为浅蓝色
ax.quiver(region_lon_skip, region_lat_skip, region_u10_skip, region_v10_skip, color='lightblue', scale=500)

# 添加海岸线和边界
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 添加经纬度网格
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 创建颜色条轴并添加颜色条
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.1, axes_class=plt.Axes)
cbar = plt.colorbar(contourf, cax=cax, orientation='vertical', label='Wind Speed (m/s)')

# 添加标题
ax.set_title('Mean Sea Level Pressure and Wind Field (10m)', loc='left')
ax.set_title(f'Max Wind Speed: {max_wind_speed:.2f} m/s  Min Pressure: {min_pressure:.2f} hPa', loc='right')

# 显示图像
plt.savefig(r'draw\mslp_and_wind.png')