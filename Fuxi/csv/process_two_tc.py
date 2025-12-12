import pandas as pd
from itertools import combinations
import numpy as np

# Haversine formula to calculate distance between two points on Earth
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def find_coexisting_typhoons(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 将 USA_PRES 列转换为数值，并剔除无法转换或缺失的记录
    df['USA_PRES'] = pd.to_numeric(df['USA_PRES'], errors='coerce')
    df = df.dropna(subset=['USA_PRES'])
    # 仅保留 USA_PRES 小于 1000 的记录
    df = df[df['USA_PRES'] < 1000]
    
    # 转换 ISO_TIME 为 datetime, coercing errors to NaT, 并剔除无效记录
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df.dropna(subset=['ISO_TIME'])
    # 筛选整六小时的记录（分钟、秒均为 0 且小时能被 6 整除）
    df = df[df['ISO_TIME'].dt.minute.eq(0) & df['ISO_TIME'].dt.second.eq(0) & (df['ISO_TIME'].dt.hour % 6 == 0)]
    
    # Filter typhoons in the Western Pacific (WP) basin
    df = df[df['BASIN'] == 'WP']
    
    # Convert LAT and LON to numeric types
    df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
    df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
    df = df.dropna(subset=['LAT', 'LON'])
    
    # Preprocess typhoon names: use SID when NAME is missing
    df['NAME'] = df['NAME'].fillna(df['SID'])
    
    # Group by SID and record each typhoon's existence times and metadata.
    typhoon_dict = {}
    for sid, group in df.groupby('SID'):
        name = group['NAME'].iloc[0]
        typhoon_dict[sid] = {
            'name': name,
            'times': set(group['ISO_TIME']),
            'data': group,  # Cache typhoon data for later use
            'initial_lat': group['LAT'].iloc[0],
            'initial_lon': group['LON'].iloc[0]
        }
    
    # Generate all possible pairs of typhoons
    sids = list(typhoon_dict.keys())
    coexisting_pairs = []
    for i in range(len(sids)):
        for j in range(i+1, len(sids)):
            sid1, sid2 = sids[i], sids[j]
            lat1, lon1 = typhoon_dict[sid1]['initial_lat'], typhoon_dict[sid1]['initial_lon']
            lat2, lon2 = typhoon_dict[sid2]['initial_lat'], typhoon_dict[sid2]['initial_lon']
            distance = haversine(lat1, lon1, lat2, lon2)
            if distance > 1500:
                continue
            common_times = typhoon_dict[sid1]['times'] & typhoon_dict[sid2]['times']
            if common_times:
                sorted_times = sorted(common_times)
                start_time = end_time = sorted_times[0]
                for t in sorted_times[1:]:
                    if (t - end_time).total_seconds() <= 3600 * 24:
                        end_time = t
                    else:
                        coexisting_pairs.append((sid1, sid2, start_time, end_time))
                        start_time = end_time = t
                coexisting_pairs.append((sid1, sid2, start_time, end_time))
    
    # Generate files for each coexisting event
    for sid1, sid2, start, end in coexisting_pairs:
        data1 = typhoon_dict[sid1]['data']
        data2 = typhoon_dict[sid2]['data']
        mask1 = (data1['ISO_TIME'] >= start) & (data1['ISO_TIME'] <= end)
        mask2 = (data2['ISO_TIME'] >= start) & (data2['ISO_TIME'] <= end)
        
        num1 = int(data1['NUMBER'].iloc[0])
        num2 = int(data2['NUMBER'].iloc[0])
        if num1 <= num2:
            ordered_data = [data1[mask1].sort_values('ISO_TIME'),
                            data2[mask2].sort_values('ISO_TIME')]
            name1 = typhoon_dict[sid1]['name'].replace(' ', '_')
            name2 = typhoon_dict[sid2]['name'].replace(' ', '_')
        else:
            ordered_data = [data2[mask2].sort_values('ISO_TIME'),
                            data1[mask1].sort_values('ISO_TIME')]
            name1 = typhoon_dict[sid2]['name'].replace(' ', '_')
            name2 = typhoon_dict[sid1]['name'].replace(' ', '_')
        
        combined_data = pd.concat(ordered_data, ignore_index=True)
        # 如果原有 WMO_PRES 存在则删除，最终仅保存 USA_PRES 改名为 WMO_PRES
        if 'WMO_PRES' in combined_data.columns:
            combined_data = combined_data.drop(columns=['WMO_PRES'])
        combined_data = combined_data.rename(columns={'USA_PRES': 'WMO_PRES'})
        output_columns = ['SID', 'SEASON', 'NUMBER', 'NAME', 'ISO_TIME',
                          'LAT', 'LON', 'WMO_PRES']
        filename = f"two-typhoon-csv\\{name1}-{name2}_{start.date()}_to_{end.date()}.csv"
        combined_data[output_columns].to_csv(filename, index=False)
        print(f"生成文件: {filename}")

if __name__ == "__main__":
    find_coexisting_typhoons(r'csv\ibtracs.last3years.list.v04r01.csv')