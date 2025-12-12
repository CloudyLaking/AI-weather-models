import netCDF4 as nc
import numpy as np

# 打开 NetCDF 文件
file_path = r'raw_data\4d1c7be19e808bd2d8a49d21920cad09.nc'
dataset = nc.Dataset(file_path, mode='r')

# 打印文件中的变量
print(dataset.variables.keys())

# 要读取的变量名列表
surface_variable_names = ['msl', 'u10', 'v10', 't2m']
upper_variable_names = ['z', 'q', 't', 'u', 'v']

# 创建一个列表来存储地表变量的数据
surface_data_list = []

for variable_name in surface_variable_names:
    if variable_name in dataset.variables:
        data = dataset.variables[variable_name][:]
        
        # 检查是否为 MaskedArray
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)  # 将掩码值填充为 NaN 或其他适当的值
        
        # 将数据添加到列表中
        surface_data_list.append(data)
        print(f"变量 '{variable_name}' 的数据已读取")
    else:
        print(f"变量 '{variable_name}' 不存在于文件中")

# 将地表变量列表转换为多维数组
surface_data_array = np.array(surface_data_list)

# 保存地表变量多维数组为 .npy 文件
np.save('input_surface.npy', surface_data_array)
print("所有地表变量的数据已保存为 input_surface.npy")

# 创建一个列表来存储高空变量的数据
upper_data_list = []

for variable_name in upper_variable_names:
    if variable_name in dataset.variables:
        data = dataset.variables[variable_name][:]
        
        # 检查是否为 MaskedArray
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)  # 将掩码值填充为 NaN 或其他适当的值
        
        # 将数据添加到列表中
        upper_data_list.append(data)
        print(f"变量 '{variable_name}' 的数据已读取")
    else:
        print(f"变量 '{variable_name}' 不存在于文件中")

# 将高空变量列表转换为多维数组
upper_data_array = np.array(upper_data_list)

# 保存高空变量多维数组为 .npy 文件
np.save('input_upper.npy', upper_data_array)
print("所有高空变量的数据已保存为 input_upper.npy")

# 关闭文件
dataset.close()