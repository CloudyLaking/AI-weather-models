import pandas as pd
import numpy as np
import os

def find_single_typhoons(csv_path):
    # 读取 CSV
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 将 USA_PRES 列转换为数值，并剔除无法转换或缺失的记录
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')
    df = df.dropna(subset=['USA_PRES'])
    
    # 仅保留 USA_PRES 小于 1000 的记录
    df = df[df['USA_PRES'] < 1000]
    
    # 转换 ISO_TIME 为 datetime，并剔除无效的时间记录
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df.dropna(subset=['ISO_TIME'])
    
    # 筛选 2018-2019 年的记录
    df = df[(df['ISO_TIME'].dt.year >= 2018) & (df['ISO_TIME'].dt.year <= 2019)]
    
    # 仅保留整六小时的记录（分钟、秒均为 0 且小时整除6）
    df = df[df['ISO_TIME'].dt.minute.eq(0) &
            df['ISO_TIME'].dt.second.eq(0) &
            (df['ISO_TIME'].dt.hour % 6 == 0)]
    
    # 转换 LAT、LON 为数值类型，并删除无效记录
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df = df.dropna(subset=['LAT', 'LON'])
    
    # 若 NAME 缺失，使用 SID 作为台风名称
    df['NAME'] = df['NAME'].fillna(df['SID'])
    
    # 按 SID 分组，按时间排序
    typhoon_dict = {}
    for sid, group in df.groupby('SID'):
        name = group['NAME'].iloc[0]
        typhoon_dict[sid] = {
            'name': name,
            'data': group.sort_values('ISO_TIME')
        }
    
    output_folder = "normal_tc"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 对每个台风，直接将所有记录输出为一个 CSV 文件
    for sid, info in typhoon_dict.items():
        data = info['data']
        # 获取该台风记录的起止时间，并扩展时间区间：起始时间前2天，结束时间后3天
        region_start = data['ISO_TIME'].min()
        region_end = data['ISO_TIME'].max()
        start_time = region_start - pd.Timedelta(days=2)
        end_time = region_end + pd.Timedelta(days=3)
        # 取扩展区间内的所有记录
        extended_data = data[(data['ISO_TIME'] >= start_time) & (data['ISO_TIME'] <= end_time)]
        # 删除原有的 WMO_PRES 列（如果存在），再将 USA_PRES 改名为 WMO_PRES
        if 'WMO_PRES' in extended_data.columns:
            extended_data = extended_data.drop(columns=['WMO_PRES'])
        extended_data = extended_data.rename(columns={'USA_PRES': 'WMO_PRES'})
        output_columns = ['SID', 'SEASON', 'NUMBER', 'NAME', 'ISO_TIME',
                          'LAT', 'LON', 'WMO_PRES']
        # 构造文件名，台风名称中的空格替换为 _
        file_name = f"{output_folder}/{info['name'].replace(' ', '_')}_{start_time.date()}_to_{end_time.date()}.csv"
        extended_data[output_columns].to_csv(file_name, index=False)
        print(f"生成文件: {file_name}")

if __name__ == "__main__":
    find_single_typhoons(r'csv\ibtracs.WP.list.v04r01.csv')