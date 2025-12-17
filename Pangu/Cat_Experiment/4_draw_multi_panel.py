# 4_draw_multi_panel.py
# 绘制六张子图 (2x3) 的全球地面温度场与风场

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================
# 配置参数
# ==========================================
TARGET_HOURS = [0, 24, 72, 360, 1446, 3606]  # 必须是6个，对应 2x3
FIG_SIZE = (11, 4.5)
# ==========================================

def find_project_root():
    """自动定位项目根目录"""
    current = os.path.dirname(os.path.abspath(__file__))
    # 向上寻找直到找到 Output 目录，或者达到系统根目录
    for _ in range(4):
        if os.path.exists(os.path.join(current, 'Output')):
            return current
        current = os.path.dirname(current)
    return os.getcwd() # Fallback

def get_file_path(hour, project_root):
    """根据时次查找对应的 .npy 文件"""
    if hour == 0:
        search_dir = os.path.join(project_root, 'Input/Pangu/Cat_Experiment')
        prefix = 'input_surface'
    else:
        search_dir = os.path.join(project_root, 'Output/Pangu/Cat_Experiment')
        prefix = 'output_surface'
    
    if not os.path.exists(search_dir):
        print(f"[WARN] Directory not found: {search_dir}")
        return None

    files = [f for f in os.listdir(search_dir) if f.startswith(prefix) and f.endswith('.npy')]
    
    if hour > 0:
        # 匹配 +00024h 这种格式
        hour_str = f"+{hour:05d}h"
        files = [f for f in files if hour_str in f]
    
    if not files:
        return None
    
    # 找最新的文件
    files.sort(key=lambda x: os.path.getmtime(os.path.join(search_dir, x)), reverse=True)
    return os.path.join(search_dir, files[0])

def load_data(file_path):
    """加载并处理数据"""
    try:
        data = np.load(file_path)
        # data shape: [4, 720, 1440] -> [mslet, u10, v10, t2m]
        u10 = data[1]
        v10 = data[2]
        t2m = data[3]
        t2m_c = t2m - 273.15 # 转摄氏度
        
        nlat, nlon = t2m.shape
        lat = np.linspace(90, -90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        lons, lats = np.meshgrid(lon, lat)
        
        return lons, lats, t2m_c, u10, v10
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return None

def draw_multi_panel():
    project_root = find_project_root()
    print(f"[INFO] Project root: {project_root}")
    
    # 创建画布 2行3列
    # central_longitude=180 实现 0-360 视图
    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE, 
                             subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    axes = axes.flatten() # 展平以便循环
    
    # 绘图参数
    vmin, vmax = -200, 200
    cmap = 'RdYlBu_r'
    step = 90 # 风羽稀疏度
    skip = (slice(None, None, step), slice(None, None, step))
    
    plot_handle = None # 用于保存最后一次绘图对象以生成色条
    init_time_str = "Unknown" # 用于主标题

    for i, hour in enumerate(TARGET_HOURS):
        ax = axes[i]
        
        # 获取文件路径
        file_path = get_file_path(hour, project_root)
        
        if file_path:
            print(f"[INFO] Plotting hour {hour} from {os.path.basename(file_path)}")
            # 尝试从文件名解析初始时间 (仅取第一个有效文件的)
            if init_time_str == "Unknown":
                try:
                    parts = os.path.basename(file_path).split('_')
                    time_part = parts[2].replace('.npy', '')
                    init_time_str = time_part.split('+')[0] if '+' in time_part else time_part
                except:
                    pass

            # 加载数据
            result = load_data(file_path)
            if result:
                lons, lats, t2m_c, u10, v10 = result
                
                # 1. 绘制温度场
                cf = ax.pcolormesh(lons, lats, t2m_c, 
                                   cmap=cmap, vmin=vmin, vmax=vmax,
                                   transform=ccrs.PlateCarree(), shading='auto')
                plot_handle = cf
                
                # 2. 绘制风羽
                ax.barbs(lons[skip], lats[skip], u10[skip], v10[skip], 
                         length=4, pivot='middle', linewidth=0.4, color='black', 
                         transform=ccrs.PlateCarree())
                
                # 统计
                t_min = np.min(t2m_c)
                t_max = np.max(t2m_c)
                
                # 子标题: (a) +24h Min: xx Max: xx
                letter = chr(97 + i) # a, b, c, d, e, f
                hour_label = f"+{hour}h" if hour > 0 else "Input(0h)"
                ax.set_title(f"({letter}) {hour_label}  Min: {t_min:.1f}°C  Max: {t_max:.1f}°C", 
                             fontsize=10, loc='left')
            else:
                ax.text(0.5, 0.5, "Data Load Error", ha='center', transform=ax.transAxes)
        else:
            print(f"[WARN] No file found for hour {hour}")
            ax.text(0.5, 0.5, f"No Data for +{hour}h", ha='center', transform=ax.transAxes)

        # 地图要素
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='black')
        ax.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.3)

    # === 全局设置 ===
    
    # 1. 主标题 (左上角，与图框左边对齐)
    fig.text(0.01, 0.98, f'Pangu Global 2m Temperature & 10m Wind', 
             ha='left', va='top', fontsize=14)

    # 2. 统一色条 (右侧)
    if plot_handle:
        # 调整子图布局，为右侧色条留出空间，减少上下间距
        plt.tight_layout(rect=[0, 0, 0.92, 0.95], pad=1.0, h_pad=1.5) 
        # 添加色条轴
        cax = fig.add_axes([0.93, 0.08, 0.015, 0.8]) # [left, bottom, width, height]
        cb = fig.colorbar(plot_handle, cax=cax, orientation='vertical')
        cb.set_label('Temperature (°C)')
        cb.set_ticks(np.arange(vmin, vmax+1, 50))
    else:
        plt.tight_layout(pad=1.0, h_pad=1.5)

    # 保存
    output_dir = os.path.join(project_root, 'Run-output-png')
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f'Multi_Panel_Temp_Wind_{init_time_str}.png'
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] Saving combined figure to: {output_path}")
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("[OK] Done.")

if __name__ == "__main__":
    draw_multi_panel()