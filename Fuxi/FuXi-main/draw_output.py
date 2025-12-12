import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math

# 定义Haversine距离计算函数
def haversine(lon1, lat1, lon2, lat2):
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # Haversine公式
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
        
        # 计算每个文件的初始时间
        initial_time = df['ISO_TIME'].min()
        
        # 计算预报时效（小时）
        df['lead_time'] = (df['ISO_TIME'] - initial_time).dt.total_seconds() / 3600
        
        # 只保留整6小时的预报数据
        df = df[df['lead_time'] % 6 == 0]
        
        # 计算预报误差
        for _, row in df.iterrows():
            # Fuxi误差计算
            if not np.isnan(row['FuxiMinLat']):
                fuxi_error = haversine(row['LON'], row['LAT'], 
                                      row['FuxiMinLon'], row['FuxiMinLat'])
                all_data.append({
                    'model': 'Fuxi',
                    'lead_time': row['lead_time'],
                    'error': fuxi_error
                })
            
            # ECMWF误差计算
            if not np.isnan(row['ECMWFMinLat']):
                ecmwf_error = haversine(row['LON'], row['LAT'], 
                                       row['ECMWFMinLon'], row['ECMWFMinLat'])
                all_data.append({
                    'model': 'ECMWF',
                    'lead_time': row['lead_time'],
                    'error': ecmwf_error
                })
    
    return pd.DataFrame(all_data)

# 主程序
folder_path = r"two-typhoon-csv\new_"
df = process_files(folder_path)

# 准备绘图数据
plt.figure(figsize=(12, 6))

# 为每个模型绘制散点和趋势线
for model in ['Fuxi', 'ECMWF']:
    model_data = df[df['model'] == model]
    
    # 绘制散点
    plt.scatter(model_data['lead_time'], model_data['error'], 
                alpha=0.5, label=f'{model} Scatter')
    
    # 计算平均值并绘制趋势线
    avg_error = model_data.groupby('lead_time')['error'].mean().reset_index()
    plt.plot(avg_error['lead_time'], avg_error['error'], 
             '-o', linewidth=2, markersize=8, label=f'{model} Mean Error')

# 设置坐标轴
plt.xlabel('Forecast Lead Time (hours)', fontsize=12)
plt.ylabel('Forecast Error (km)', fontsize=12)
plt.title('Typhoon Track Forecast Error', fontsize=14)

# 设置x轴刻度（6小时间隔）
max_lead = int(df['lead_time'].max())
plt.xticks(np.arange(6, max_lead+6, 6))

# 添加图例和网格
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# 显示图表
plt.show()