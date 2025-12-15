# AIFS 工作流 - 快速参考

## 一键运行完整工作流

```bash
cd AIFS/Main
python main_aifs_workflow.py
```

## 命令行选项

```bash
# 使用最新ECMWF数据
python main_aifs_workflow.py

# 指定起报时间
python main_aifs_workflow.py --datetime 2025121200

# 指定预报时效
python main_aifs_workflow.py --lead-time 24

# 使用CPU计算
python main_aifs_workflow.py --device cpu

# 指定绘制的气象要素
python main_aifs_workflow.py --fields 2t 100u msl gh_500 t_850

# 跳过可视化
python main_aifs_workflow.py --no-draw

# 完整示例
python main_aifs_workflow.py --datetime 2025121200 --lead-time 12 --device cuda --fields 2t 100u msl
```

## Python API 使用

### 仅获取数据
```python
from get_data_aifs import get_aifs_data

input_state = get_aifs_data()
print(f"Date: {input_state['date']}")
print(f"Fields: {list(input_state['fields'].keys())}")
```

### 数据 + 预报
```python
from get_data_aifs import get_aifs_data
from run_aifs import AIFSRunner

input_state = get_aifs_data()
runner = AIFSRunner(device='cuda')
output_files = runner.run_forecast(input_state, lead_time=12)
```

### 完整工作流
```python
from main_aifs_workflow import run_complete_aifs_workflow

results = run_complete_aifs_workflow(
    lead_time=12,
    device='cuda',
    draw_results=True
)

print(f"Status: {results['status']}")
print(f"Images generated: {len(results['image_files'])}")
```

### 自定义可视化
```python
from run_aifs import AIFSRunner
from draw_aifs_results import draw_aifs_results
from get_data_aifs import get_aifs_data

input_state = get_aifs_data()
runner = AIFSRunner()

for state in runner.runner.run(input_state=input_state, lead_time=12):
    images = draw_aifs_results(
        state,
        fields_to_draw=['2t', '100u', 'msl', 'gh_500'],
        draw_wind=True
    )
```

## 输出文件位置

| 类型 | 目录 | 示例 |
|------|------|------|
| 原始数据 | `Input/AIFS_raw/` | (NetCDF) |
| 处理数据 | `Input/AIFS/` | input_aifs.npz |
| 预报输出 | `Output/AIFS/` | output_aifs_2025121200+012h.npz |
| 气象图 | `Run-output-png/AIFS/` | aifs_2t_2025121200+012h.png |

## 常用气象要素

| 字段 | 描述 | 单位 |
|------|------|------|
| 2t | 2米气温 | K |
| 100u, 100v | 100米风 | m/s |
| msl | 平均海平面气压 | Pa |
| sp | 地表气压 | Pa |
| tcw | 大气可降水量 | kg/m² |
| z_XXX | 地位势（XXX=气压） | m²/s² |
| t_XXX | 温度（XXX=气压） | K |
| u_XXX, v_XXX | 风分量（XXX=气压） | m/s |
| q_XXX | 比湿（XXX=气压） | kg/kg |
| w_XXX | 竖直速度（XXX=气压） | m/s |

常用气压层: 1000, 925, 850, 700, 500, 300, 200, 100 hPa

## 支持的计算设备

- `cuda` - NVIDIA GPU（推荐，最快）
- `cpu` - 中央处理器（备用，较慢）

## 环境变量优化

对于大GPU内存：
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

对于有限GPU内存（12-16GB）：
```bash
export ANEMOI_INFERENCE_NUM_CHUNKS=16
```

在Python中设置：
```python
import os
os.environ['ANEMOI_INFERENCE_NUM_CHUNKS'] = '16'

from run_aifs import AIFSRunner
runner = AIFSRunner(num_chunks=16)
```

## 故障排除

### 导入错误
```
[ERROR] earthkit package not installed
```
→ 运行: `pip install earthkit-data earthkit-regrid`

### GPU内存不足
→ 使用: `AIFSRunner(num_chunks=16)` 或 `--device cpu`

### 无法连接ECMWF Open Data
→ 检查网络连接，或使用已下载的缓存数据

### 预报输出为空
→ 检查初始条件是否正确加载，查看日志消息

## 性能参考

| 设备 | 12h预报 | 24h预报 |
|------|---------|---------|
| NVIDIA A100 | ~2-3分钟 | ~4-5分钟 |
| NVIDIA RTX 4090 | ~5-8分钟 | ~10-15分钟 |
| CPU (16核) | ~20-30分钟 | ~40-60分钟 |

## 文件架构

```
AIFS/Main/
├── get_data_aifs.py          # 数据获取模块
├── run_aifs.py               # 模型执行模块
├── draw_aifs_results.py      # 可视化模块
├── main_aifs_workflow.py     # 工作流编排
└── AIFS_README.md            # 详细文档
```

所有模块与Pangu架构对标，可互换使用。

## 数据流程

```
ECMWF Open Data
      ↓
get_data_aifs.py (处理为N320分辨率)
      ↓
Input/AIFS/
      ↓
run_aifs.py (12小时预报)
      ↓
Output/AIFS/
      ↓
draw_aifs_results.py (生成气象图)
      ↓
Run-output-png/AIFS/
```

---

**提示**: 首次运行时，数据下载和模型初始化可能需要5-10分钟。后续运行会使用缓存，速度更快。
