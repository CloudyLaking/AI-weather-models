# 图像转气象场初始条件程序
# 根据论文中的方法：X_v(x,y) = a*I(x,y) + b
# 其中 I(x,y) 是灰度值，映射到 [-1000, 1000] 范围

import os
import sys
import numpy as np
from PIL import Image
from datetime import datetime

def load_and_convert_image(image_path, target_shape=(721, 1440)):
    """
    加载图像并转换为气象场初始条件
    
    Args:
        image_path: 图像文件路径
        target_shape: 目标形状 (height, width)
        
    Returns:
        tuple: (input_surface, input_upper) NPY数组
    """
    print(f"[INFO] Loading image from: {image_path}")
    
    try:
        # 加载图像
        img = Image.open(image_path)
        print(f"[OK] Image loaded, size: {img.size}")
        
        # 转换为灰度图
        if img.mode != 'L':
            img = img.convert('L')
        
        # 转换为numpy数组
        img_array = np.array(img, dtype=np.float32)
        print(f"[OK] Converted to grayscale, shape: {img_array.shape}")
        
        # 验证大小是否为 720x1440 (目标 721x1440)
        if img_array.shape == (720, 1440):
            # 图像是720x1440，需要补一行到721x1440
            print(f"[INFO] Image size 720×1440, adding one row from top...")
            first_row = img_array[0:1, :]  # 复制第一行(另一边)
            img_array = np.vstack([img_array, first_row])  # 添加到末尾
            print(f"[OK] Extended to: {img_array.shape}")
        elif img_array.shape != target_shape:
            print(f"[WARN] Image size {img_array.shape} != expected {target_shape}")
            print(f"       Resizing image...")
            img_resized = Image.fromarray(img_array.astype(np.uint8))
            img_resized = img_resized.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32)
            print(f"[OK] Resized to: {img_array.shape}")
        
        # 归一化灰度值到 [0, 1]
        # 为了确保映射后范围正好是 [-1000, 1000]，使用图像实际的最值进行归一化
        img_min = img_array.min()
        img_max = img_array.max()
        print(f"[INFO] Original image value range: [{img_min}, {img_max}]")
        
        if img_max > img_min:
            img_normalized = (img_array - img_min) / (img_max - img_min)
        else:
            img_normalized = img_array / 255.0
            
        print(f"[OK] Normalized grayscale range: [{img_normalized.min():.4f}, {img_normalized.max():.4f}]")
        
        # 将灰度值映射到 [-1000, 1000]
        # 公式: X_v = a*I + b，其中 I ∈ [0, 1]，X_v ∈ [-1000, 1000]
        # 当 I=0 时，X_v=-1000; 当 I=1 时，X_v=1000
        # 因此: a=2000, b=-1000
        a = 2000.0
        b = -1000.0
        img_scaled = a * img_normalized + b
        
        print(f"[OK] Mapped to physical range [-1000, 1000]")
        print(f"     Range: [{img_scaled.min():.2f}, {img_scaled.max():.2f}]")
        
        # ===== 构造Pangu输入数据 =====
        # Surface: [4, 720, 1440] -> [mslet, u10, v10, t2m]
        # Upper: [5, 13, 720, 1440] -> [gh, q, t, u, v]
        # 所有变量统一映射到 [-1000, 1000]，不考虑物理意义
        
        # 地面数据：所有4个变量都直接使用缩放后的图像值
        input_surface = np.stack([
            img_scaled,  # mslet
            img_scaled,  # u10
            img_scaled,  # v10
            img_scaled   # t2m
        ], axis=0).astype(np.float32)
        
        print(f"[OK] Surface layer created, shape: {input_surface.shape}")
        print(f"     All variables: [{img_scaled.min():.2f}, {img_scaled.max():.2f}]")
        
        # 上层大气数据: [5, 13, 720, 1440]
        # 5 variables: gh, q, t, u, v
        # 13 pressure levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50
        # 所有变量和所有高度都使用相同的缩放图像
        input_upper = np.zeros((5, 13, target_shape[0], target_shape[1]), dtype=np.float32)
        
        # 为每个变量和每个高度层分配相同的缩放图像值
        for v in range(5):  # 5 variables: gh, q, t, u, v
            for k in range(13):  # 13 pressure levels
                input_upper[v, k] = img_scaled
        
        print(f"[OK] Upper layer created, shape: {input_upper.shape}")
        print(f"     All variables & levels: [{img_scaled.min():.2f}, {img_scaled.max():.2f}]")
        
        return input_surface, input_upper
        
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主程序"""
    
    # 文件路径
    image_path = os.path.join(
        os.path.dirname(__file__), 
        '04afd9a6bf31bb26af577106b6fefb0f.jpg'
    )
    
    output_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__),
        '../../../Input/Pangu/Cat_Experiment'
    ))
    
    print("\n" + "="*70)
    print("IMAGE TO INITIAL FIELD CONVERSION")
    print("="*70)
    print(f"[START] Conversion started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换图像
    input_surface, input_upper = load_and_convert_image(image_path)
    
    if input_surface is None or input_upper is None:
        print("[ERROR] Conversion failed")
        return False
    
    # 保存NPY文件
    try:
        surface_path = os.path.join(output_dir, 'input_surface.npy')
        upper_path = os.path.join(output_dir, 'input_upper.npy')
        
        np.save(surface_path, input_surface)
        np.save(upper_path, input_upper)
        
        print(f"\n[OK] Files saved successfully:")
        print(f"     Surface: {surface_path}")
        print(f"     Upper:   {upper_path}")
        
        # 打印统计信息
        print(f"\n[STATS] Surface data (4 variables):")
        var_names = ['mslet', 'u10', 'v10', 't2m']
        for i, var_name in enumerate(var_names):
            print(f"  ✓ {var_name}: min={input_surface[i].min():.4f}, max={input_surface[i].max():.4f}")
        
        print(f"\n[STATS] Upper data (5 variables × 13 levels):")
        var_names = ['gh', 'q', 't', 'u', 'v']
        for i, var_name in enumerate(var_names):
            print(f"  ✓ {var_name}: min={input_upper[i].min():.4f}, max={input_upper[i].max():.4f}")
        
        print(f"\n[OK] Conversion completed successfully!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save files: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
