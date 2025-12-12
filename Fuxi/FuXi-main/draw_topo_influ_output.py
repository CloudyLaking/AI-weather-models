import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

# 定义Haversine距离计算函数
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # 地球平均半径，单位公里
    return c * r

# 读取并处理数据
def process_files(folder_path):
    all_data = []
    
    for file in glob.glob(os.path.join(folder_path, "*.csv")):
        print(f"Processing file: {file}")   
        df = pd.read_csv(file)
        
        # 转换时间格式
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
        
        # 筛选留下气压低于990 hPa的记录（任一模型低于阈值即可）
        pressure_threshold = 990
        df = df[(df['FuXiMinPressure'] < pressure_threshold) | (df['ECMWFMinPressure'] < pressure_threshold)]
        
        # 如果同一时刻同一位置出现多个记录，则过滤掉这些重复数据
        df = df.groupby(['ISO_TIME', 'LON', 'LAT']).filter(lambda grp: len(grp) == 1)
        
        # 计算每个文件的初始时间
        initial_time = df['ISO_TIME'].min()
        
        # 计算预报时效（小时）
        df['lead_time'] = (df['ISO_TIME'] - initial_time).dt.total_seconds() / 3600
        
        # 只保留整6小时的预报数据
        df = df[df['lead_time'] % 6 == 0]
        
        # 计算预报误差，并记录数据来源（同时保存标准位置：LON, LAT）
        for _, row in df.iterrows():
            # FuXi误差计算
            if not np.isnan(row['FuXiMinLat']):
                fuxi_error = haversine(row['LON'], row['LAT'], row['FuXiMinLon'], row['FuXiMinLat'])
                all_data.append({
                    'model': 'FuXi',
                    'lead_time': row['lead_time'],
                    'error': fuxi_error,
                    'source': file,
                    'LON': row['LON'],
                    'LAT': row['LAT']
                })
            
            # ECMWF误差计算
            if not np.isnan(row['ECMWFMinLat']):
                ecmwf_error = haversine(row['LON'], row['LAT'], row['ECMWFMinLon'], row['ECMWFMinLat'])
                all_data.append({
                    'model': 'ECMWF',
                    'lead_time': row['lead_time'],
                    'error': ecmwf_error,
                    'source': file,
                    'LON': row['LON'],
                    'LAT': row['LAT']
                })
    
    return pd.DataFrame(all_data)

# 主程序
folder_path = r"topo-influ-csv\new_"
df = process_files(folder_path)

# 剔除每个文件中最后一个数据点（每个 source+model 分组中 lead_time 最大的记录）
df_filtered = df.groupby(['source', 'model'], group_keys=False).apply(
    lambda g: g[g['lead_time'] != g['lead_time'].max()]
).reset_index(drop=True)

# 设置时效限制，只保留 120h 内的记录
df_filtered = df_filtered[df_filtered['lead_time'] <= 120]

# 过滤掉“第一个时次误差大于100 km”的组合
df_filtered = df_filtered.groupby(['source', 'model'], group_keys=False).filter(
    lambda g: g.loc[g['lead_time'].idxmin(), 'error'] <= 100
)

# 打印误差超过1000 km 的记录及其数据来源
print("误差超过1000 km 的记录：")
print(df_filtered[df_filtered['error'] > 1000])

# 绘制预报误差-时效图
plt.figure(figsize=(12, 6))
for model in ['FuXi', 'ECMWF']:
    model_data = df_filtered[df_filtered['model'] == model]
    plt.scatter(model_data['lead_time'], model_data['error'], 
                alpha=0.5, label=f'{model} Scatter')
    avg_error = model_data.groupby('lead_time')['error'].mean().reset_index()
    plt.plot(avg_error['lead_time'], avg_error['error'], 
             '-o', linewidth=2, markersize=8, label=f'{model} Mean Error')
plt.xlabel('Forecast Lead Time (hours)', fontsize=12)
plt.ylabel('Forecast Error (km)', fontsize=12)
plt.title('Typhoon Track Forecast Error', fontsize=14)
max_lead = int(df_filtered['lead_time'].max())
plt.xticks(np.arange(6, max_lead+6, 6))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


'''
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.figure(figsize=(10,8))
ax = plt.axes(projection=ccrs.PlateCarree())
# 根据数据范围设置地图显示范围，可根据需要调整边界偏移量
ax.set_extent([df_filtered['LON'].min()-2, df_filtered['LON'].max()+2, 
               df_filtered['LAT'].min()-2, df_filtered['LAT'].max()+2], crs=ccrs.PlateCarree())
# 绘制海岸线及国家边界
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black', alpha=0.2)
# 绘制误差散点：使用 vmin=0, vmax=400 设置颜色映射范围
sc = ax.scatter(df_filtered['LON'], df_filtered['LAT'], c=df_filtered['error'],
                cmap='viridis', s=30, edgecolors='k', alpha=0.9, 
                vmin=0, vmax=300, transform=ccrs.PlateCarree())
plt.colorbar(sc, ax=ax, label="Forecast Error (km)")
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
plt.title('Error Distribution Map with Coastlines', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
'''