import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm

# 指定 .npy 文件的路径
file_path = 'output_surface.npy'

# 加载 .npy 文件
data = np.load(file_path)

# 提取2米温度数据并转换为摄氏度
t2m = data[3] - 273.15

# 创建经纬度网格
lon = np.linspace(0, 360, t2m.shape[1])
lat = np.linspace(90, -90, t2m.shape[0])
lon, lat = np.meshgrid(lon, lat)

# 华东地区的经纬度范围
lon_min, lon_max = 113, 130
lat_min, lat_max = 24, 40

# 提取华东地区的索引范围
lon_indices = np.where((lon[0, :] >= lon_min) & (lon[0, :] <= lon_max))[0]
lat_indices = np.where((lat[:, 0] >= lat_min) & (lat[:, 0] <= lat_max))[0]

# 提取华东地区的温度数据
east_china_t2m = t2m[np.ix_(lat_indices, lon_indices)]
east_china_lon = lon[np.ix_(lat_indices, lon_indices)]
east_china_lat = lat[np.ix_(lat_indices, lon_indices)]

# 创建绘图
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())

# 自定义色阶，体现温度冷暖
colors = [
    '#BC00D1FF', '#D359FFFF', '#8B66DFFF', '#817CEDFF', '#6690FAFF', '#11B8FFFF', 
    '#65D8FFFF', '#42FFECFF', '#00FFB3FF', '#A4FFDCFF', '#FFFFFFFF', '#C3FFCEFF', 
    '#9BFF9BFF', '#4DFF62FF', '#00FF11FF', '#59FF00FF', '#00CD0EFF', '#14B91AFF',
    '#54BD09FF', '#127631FF', '#80A52AFF', '#95BA0EFF', '#C9C01AFF', '#E1E723FF', 
    '#D9DC11FF', '#FEEF10FF', '#FFC547FF', '#FFAE00FF', '#FF8F06FF', '#FFA07A', 
    '#FFA500', '#FFB6C1', '#FFC0CB', '#FFD700', '#FFDAB9', '#FFDEAD', 
    '#FF814FFF', '#FF5500FF', '#DF512AFF', '#C26400FF', '#8B4606FF', '#4A0000FF', 
    '#AB1919FF', '#C90A0AFF', '#D30000FF', '#FF0000FF', '#FF3131FF', '#C95D5DFF', 
    '#9E5A5AFF', '#665959FF', '#111111FF'
]
cmap = ListedColormap(colors)

# 定义边界
bounds = list(range(-30, 0, 5)) + list(range(0, 42, 2)) + list(range(42, 52, 5))
norm = BoundaryNorm(bounds, cmap.N)

# 绘制华东地区的2米温度数据，使用自定义色阶
mesh = ax.pcolormesh(east_china_lon, east_china_lat, east_china_t2m, cmap=cmap, norm=norm, shading='auto', transform=ccrs.PlateCarree())

# 添加海岸线和边界
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# 添加经纬度网格
gl = ax.gridlines(draw_labels=True, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# 添加颜色条
cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, boundaries=bounds, ticks=bounds)
cbar.set_label('2m Temperature (°C)')

# 添加标题
plt.title('2m Temperature Distribution in East China (°C)')

# 显示图像
plt.savefig('east_china_temperature2.png')