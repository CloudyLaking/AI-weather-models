# AIFS 天气预报模型完整工作流 - 使用指南

本目录包含四个Python模块，完整实现AIFS天气预报模型的数据获取、模型预报和结果可视化流程。

## 文件概览

### 1. `get_data_aifs.py` - 数据下载和转换模块
**功能**: 从 ECMWF Open Data 获取初始条件并转换为NPY格式

**主要类**:
- `AIFSDataDownloader` - 数据下载和转换类
  - `get_latest_date()` - 获取最新可用的ECMWF数据日期
  - `get_open_data()` - 从ECMWF Open Data获取并插值数据到N320分辨率
  - `process_ecmwf_data()` - 完整的数据处理流程

**支持的参数**:
- 地面参数: 10u, 10v, 2d, 2t, msl, skt, sp, tcw, lsm, z, slor, sdor
- 土壤参数: vsw, sot（自动重映射为swvl1/2, stl1/2）
- 气压层参数: gh(→z), t, u, v, w, q（13层：1000-50 hPa）

**使用示例**:
```python
from get_data_aifs import get_aifs_data

# 获取最新ECMWF数据
input_state = get_aifs_data()

# 或指定具体日期
input_state = get_aifs_data(datetime_str='2025121200')
```

---

### 2. `run_aifs.py` - 模型执行模块
**功能**: 加载AIFS模型检查点（来自Hugging Face）并执行天气预报

**主要类**:
- `AIFSRunner` - AIFS执行器
  - `run_forecast()` - 执行预报并保存输出
  - `load_forecast_state()` - 读取保存的预报状态
  - `_save_forecast_state()` - 保存预报状态为NPZ文件

**模型信息**:
- 使用 `anemoi-inference` 和 `anemoi-models` 包
- Hugging Face检查点: `ecmwf/aifs-single-1.0`
- 支持GPU（CUDA）和CPU计算
- 可配置内存优化（分块处理）

**使用示例**:
```python
from get_data_aifs import get_aifs_data
from run_aifs import run_aifs_forecast

# 获取初始条件
input_state = get_aifs_data()

# 执行12小时预报
output_files = run_aifs_forecast(
    input_state,
    lead_time=12,
    device='cuda'
)
```

---

### 3. `draw_aifs_results.py` - 结果可视化模块
**功能**: 绘制AIFS预报结果的专业气象图

**主要类**:
- `AIFSResultDrawer` - 绘图类
  - `draw_field_contourf()` - 绘制单个字段的填色等高线图（使用三角剖分）
  - `draw_wind_field()` - 绘制风场（风速填色+风矢）
  - `fix_longitudes()` - 经度坐标转换（0-360 ↔ -180-180）

**绘图特性**:
- 使用三角剖分处理N320网格的不规则分布
- 支持多种配色方案（RdBu_r等）
- 自定义风速分级和颜色映射
- 自动生成PNG格式输出

**使用示例**:
```python
from run_aifs import AIFSRunner
from draw_aifs_results import draw_aifs_results

# 执行预报并绘制结果
runner = AIFSRunner()
for state in runner.runner.run(input_state=input_state, lead_time=12):
    image_files = draw_aifs_results(
        state,
        init_datetime_str='2025121200',
        fields_to_draw=['2t', '100u', 'msl']
    )
```

---

### 4. `main_aifs_workflow.py` - 完整工作流编排脚本
**功能**: 整合数据获取、模型预报、结果可视化的完整自动化流程

**主要函数**:
- `run_complete_aifs_workflow()` - 执行完整工作流
- `main()` - 命令行接口

**工作流步骤**:
1. STEP 1: 从ECMWF Open Data获取初始条件
2. STEP 2: 使用AIFS模型执行天气预报
3. STEP 3: 绘制预报结果的气象图

**使用示例**:

使用最新ECMWF数据：
```bash
python main_aifs_workflow.py
```

指定起报时间：
```bash
python main_aifs_workflow.py --datetime 2025121200
```

使用CPU计算：
```bash
python main_aifs_workflow.py --device cpu
```

指定预报时效和绘制字段：
```bash
python main_aifs_workflow.py --lead-time 24 --fields 2t 100u msl gh_500
```

跳过可视化：
```bash
python main_aifs_workflow.py --no-draw
```

---

## 文件结构说明

```
AIFS/Main/
├── get_data_aifs.py           # 数据模块
├── run_aifs.py                # 模型执行模块
├── draw_aifs_results.py       # 可视化模块
├── main_aifs_workflow.py      # 工作流编排
└── run_AIFS_v1.ipynb          # 参考Notebook
```

**输出目录结构**:
```
Input/AIFS/              # 处理后的输入数据（NPY/NPZ）
Input/AIFS_raw/         # 原始数据（NetCDF）
Output/AIFS/            # 预报输出（NPZ）
Run-output-png/AIFS/    # 可视化结果（PNG）
```

---

## 依赖包

### 核心依赖
```
anemoi-inference>=0.4.9
anemoi-models>=0.3.1
torch>=2.4.0
ecmwf-opendata
earthkit-data>=0.3.0
earthkit-regrid>=0.4.0
```

### 可视化依赖
```
matplotlib
cartopy
numpy
scipy
```

### 安装命令
```bash
# 核心包
pip install anemoi-inference[huggingface]==0.4.9 anemoi-models==0.3.1 torch==2.4.0
pip install earthkit-data earthkit-regrid ecmwf-opendata

# 可视化包
pip install matplotlib cartopy scipy

# 可选：GPU加速
pip install flash_attn
```

---

## 关键参数和字段

### AIFS 支持的气象要素

**地面层参数** (12个):
- Wind: 10u, 10v
- Temperature: 2d (露点温度), 2t (2米气温)
- Pressure: msl (平均海平面气压), sp (地表气压)
- Moisture: tcw (大气可降水量)
- Surface: skt (地表温度), lsm (陆海掩膜)
- Orography: z (地形高度), slor (斜率), sdor (标准偏差）

**土壤层参数** (4个):
- swvl1, swvl2 (土壤含水量 第1-2层)
- stl1, stl2 (土壤温度 第1-2层)

**气压层参数** (65个 = 5参数 × 13层):
- Variables: z (地位势), t (温度), u (东西风), v (南北风), q (比湿), w (竖直速度)
- Levels: 1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa

### 坐标系统
- 水平分辨率: N320 (64×128 网格)
- 时间分辨率: 可配置（支持任意小时数）
- 坐标: 均匀网格（Gaussian网格）

---

## 常见问题

**Q: 如何在没有GPU的情况下运行？**
A: 使用 `--device cpu` 参数，但预报速度会显著降低。

**Q: 预报输出的格式是什么？**
A: NPZ压缩格式，包含日期、纬度、经度和所有气象字段。

**Q: 如何只运行数据获取而不进行预报？**
A: 直接使用 `get_data_aifs.py` 中的 `get_aifs_data()` 函数。

**Q: 内存占用过高？**
A: 在初始化 `AIFSRunner` 时设置 `num_chunks` 参数（例如16）来启用分块处理。

**Q: 无法连接到ECMWF Open Data？**
A: 检查网络连接，或使用缓存的数据（如果已下载过）。

---

## 性能指标

在标准GPU（如NVIDIA A100）上的典型执行时间：

| 步骤 | 耗时 | 备注 |
|------|------|------|
| 数据获取 | 2-5分钟 | 首次下载，后续使用缓存 |
| 模型预报 (12h) | 2-3分钟 | GPU上，单GPU运行 |
| 可视化 | 1-2分钟 | 取决于绘制字段数量 |
| **总计** | **5-10分钟** | 首次运行 |

---

## 脚本设计特点

✅ **与Pangu架构完全对标**: 采用相同的文件结构、类设计和接口规范

✅ **完整的错误处理**: 详细的日志输出和异常管理

✅ **灵活的配置**: 命令行参数支持所有主要选项

✅ **中文注释+英文标签**: 代码注释用中文，气象图标签用英文

✅ **模块化设计**: 每个模块可独立使用或组合调用

✅ **自动路径转换**: 相对路径自动转换为绝对路径

✅ **进度信息输出**: 清晰的 [INFO] [OK] [ERROR] 标记

✅ **跳过现有文件**: 支持断点续传和增量更新

---

## 注意事项

1. **首次运行时间**: 首次下载ECMWF数据可能需要5-10分钟，请耐心等待
2. **网络要求**: 需要稳定的互联网连接以访问ECMWF Open Data和Hugging Face
3. **GPU内存**: 建议至少12GB显存，可通过 `num_chunks` 参数优化内存占用
4. **数据保留**: 原始NetCDF数据会被保存用于缓存，可手动删除以节省空间

---

## 许可证

参考项目根目录 LICENSE 文件

---

**最后更新**: 2025年12月15日
