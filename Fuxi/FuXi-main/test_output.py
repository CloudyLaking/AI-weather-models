import xarray as xr
import numpy as np

# 读取 NetCDF 文件
file_path = r"output/090.nc"
data = xr.open_dataset(file_path)

# 提取层次和变量数据
level = data['level'].values
value = data['__xarray_dataarray_variable__']

# 找到 MSL 的索引
msl_index = np.where(level == 'MSL')[0][0]

# 提取对应层次的数据
pressure_data = value[0, 0, msl_index, :, :].values

# 获取经纬度数据
lat = data['lat'].values
lon = data['lon'].values

# 输出验证
print("气压数据形状:", pressure_data.shape)
print(pressure_data)
print("经度范围:", lon.min(), lon.max())
print("纬度范围:", lat.min(), lat.max())

# 关闭数据集
data.close()

import matplotlib.pyplot as plt

# 筛选区域：北纬 0° 到 40°，东经 110° 到 175°
lat_mask = (lat >= 0) & (lat <= 40)
lon_mask = (lon >= 110) & (lon <= 175)
lat_region = lat[lat_mask]
lon_region = lon[lon_mask]
pressure_region = pressure_data[np.ix_(lat_mask, lon_mask)]

plt.figure(figsize=(10, 6))
plt.contourf(lon_region, lat_region, pressure_region, cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('0-40N, 110-175E')
plt.show()