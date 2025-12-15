# AI-weather-models 工作区结构说明

## 工作区概述
本工作区根目录位于 `c:\Users\lyz13\OneDrive\Desktop\AI-weather-models\`，是一个集成多个 AI 气象模型的统一环境。

**重点**：
- 大 AI-weather-models 是工作区根目录
- 小 AI-weather-models/ 文件夹包含项目源代码（Pangu、Fuxi、AIFS 等）
- Input、Output、Models-weights、Run-output-png 等数据目录与小 AI-weather-models 文件夹**平行**，位于工作区根目录

工作区暂时包含三个主要天气预报模型：
- **Pangu**：小时级别天气预报模型      ECMWF\GFS驱动还有问题
- **Fuxi**：中期天气预报模型  
- **AIFS**：欧洲中期天气预报中心的 AI 模型   

## 工作区整体结构

```
c:\Users\lyz13\OneDrive\Desktop\AI-weather-models\  (工作区根目录)
│
├─ AI-weather-models/                   # 项目源代码放在这里
│  ├─ Pangu/
│  │  ├─ AImodels-raw/                  # 原始老代码
│  │  ├─ Main/                          # Pangu 核心代码
│  │  │  ├─ get-data-pangu.py           # 数据获取模块
│  │  │  ├─ run-pangu.py                # 模型执行模块
│  │  │  ├─ draw-pangu-results.py       # 结果可视化模块
│  │  │  └─ main-pangu-workflow.py      # 工作流集成模块
│  │  ├─ model/                         # 其他 Pangu 相关文件
│  │  ├─ input/                         # 临时文件
│  │  └─ ...
│  ├─ Fuxi/                             # Fuxi 项目文件夹
│  │  ├─ Main/                          # Fuxi 核心代码
│  │  ├─ FuXi-main/                     # Fuxi 主要实现
│  │  ├─ requirements.txt
│  │  ├─ csv/                           # CSV 数据文件
│  │  └─ ...
│  ├─ AIFS/                             # AIFS 项目文件夹
│  │  ├─ draw-aifs-from-open-store/     # AIFS 可视化代码
│  │  │  └─ draw_aifs_snow.py           # AIFS 降雪可视化脚本
│  │  ├─ Main/
│  │  └─ ...
│  ├─ aifs-open-data/                   # 开源 AIFS 数据存储（GRIB2 格式）
│  ├─ aifs-open-data-output-png/        # 开源 AIFS 可视化输出
│  ├─ README.md
│  └─ LICENSE
│
├─ Input/                                # 输入数据目录（按模型分类）
│  ├─ Pangu/                            # Pangu 处理后的模型输入数据（NPY格式）
│  │  ├─ input_upper.npy                # 高层风、温度等
│  │  └─ input_surface.npy              # 地表气压、风、温度等
│  ├─ Pangu_raw/                        # Pangu 原始下载数据（NC/GRIB格式）
│  │  ├─ era5_surface_*.nc              # ERA5 地表数据（NC格式）
│  │  ├─ era5_upper_*.nc                # ERA5 高层数据（NC格式）
│  │  ├─ gfs_*.grib2                    # GFS 数据（GRIB2格式）
│  │  └─ ecmwf_*.grib                   # ECMWF 数据（GRIB格式）
│  ├─ Fuxi/                             # Fuxi 处理后的模型输入数据
│  ├─ Fuxi_raw/                         # Fuxi 原始下载数据
│  ├─ AIFS/                             # AIFS 处理后的模型输入数据
│  └─ AIFS_raw/                         # AIFS 原始下载数据
│
├─ Output/                               # 模型预报输出数据目录（按模型分类）
│  ├─ Pangu/                            # Pangu 预报输出（NPY 格式）
│  │  ├─ output_upper_*.npy             # 高层预报结果
│  │  └─ output_surface_*.npy           # 地表预报结果
│  ├─ Fuxi/                             # Fuxi 预报输出
│  └─ AIFS/                             # AIFS 预报输出
│
├─ Models-weights/                       # 神经网络模型权重文件目录（按模型分类）
│  ├─ Pangu/                            # Pangu ONNX 模型
│  │  ├─ pangu_weather_6.onnx           # 6小时预报模型
│  │  └─ pangu_weather_24.onnx          # 24小时预报模型
│  ├─ Fuxi/                             # Fuxi 模型权重
│  │  ├─ short/                         # 短期预报
│  │  ├─ medium/                        # 中期预报
│  │  └─ long/                          # 长期预报
│  └─ AIFS/                             # AIFS 模型权重
│
└─ Run-output-png/                       # 可视化结果输出目录（PNG 图片，按模型分类）
   ├─ Pangu/                            # Pangu 可视化结果
   │  ├─ pangu_mslp_wind_*.png          # MSLP 和风场图
   │  └─ pangu_temperature_*.png        # 气温分布图
   ├─ Fuxi/                             # Fuxi 可视化结果
   └─ AIFS/                             # AIFS 可视化结果
```

## 关键目录说明

### 1. Input/ - 模型输入数据目录
所有模型的输入数据按模型存放于此（位于工作区根目录），包括原始下载数据和处理后数据：

#### Input/Pangu/ - 处理后的输入数据
Pangu 模型使用的处理后的输入数据（NPY 格式）：
- `input_upper.npy`：高层数据（风、温度、位势高度等）
- `input_surface.npy`：地表数据（气压、风、温度、湿度等）

**产生方式**：
1. 原始数据从外部 API 下载到 `Input/Pangu_raw/`
2. 由 `AI-weather-models/Pangu/Main/get-data-pangu.py` 转换为 NPY 格式
3. 最终保存在 `Input/Pangu/`

#### Input/Pangu_raw/ - 原始下载数据
从气象数据源直接下载的原始文件（NC/GRIB 格式）：
- `era5_surface_*.nc`：ERA5 地表再分析数据（NetCDF 格式）
- `era5_upper_*.nc`：ERA5 高层再分析数据（NetCDF 格式）
- `gfs_*.grib2`：GFS 预报数据（GRIB2 格式）
- `ecmwf_*.grib`：ECMWF 预报数据（GRIB 格式）

**用途**：临时存储，用于格式转换。转换后可删除。

#### 其他模型的数据目录
- **Input/Fuxi/** & **Input/Fuxi_raw/**：Fuxi 模型的处理后和原始数据
- **Input/AIFS/** & **Input/AIFS_raw/**：AIFS 模型的处理后和原始数据

### 2. Output/ - 模型预报输出数据目录
模型推理（预报运算）生成的原始数据结果（位于工作区根目录）：
- **Output/Pangu/**：
  - `output_upper_*.npy`：高层预报结果
  - `output_surface_*.npy`：地表预报结果
- **Output/Fuxi/**：Fuxi 预报输出
- **Output/AIFS/**：AIFS 预报输出

**产生方式**：由 `AI-weather-models/Pangu/Main/run-pangu.py` 等模型执行模块运行神经网络生成

### 3. Models-weights/ - 神经网络模型权重目录
存放所有已训练的模型权重文件（二进制格式，位于工作区根目录）：
- **Models-weights/Pangu/**：
  - `pangu_weather_6.onnx`：6 小时预报 ONNX 模型
  - `pangu_weather_24.onnx`：24 小时预报 ONNX 模型
  - 还有1、3小时可以放置
- **Models-weights/Fuxi/**：多时效 Fuxi 模型
- **Models-weights/AIFS/**：AIFS 模型权重

**用途**：由 `AI-weather-models/Pangu/Main/run-pangu.py` 等模块加载这些权重文件进行推理预报

### 4. Run-output-png/ - 可视化结果输出目录
模型预报结果的可视化图片输出（PNG 格式，位于工作区根目录）：
- **Run-output-png/Pangu/**：
  - `pangu_mslp_wind_*.png`：海平面气压 + 10m 风场图
  - `pangu_temperature_*.png`：2m 气温分布图
- **Run-output-png/Fuxi/**：Fuxi 可视化结果
- **Run-output-png/AIFS/**：AIFS 可视化结果

**产生方式**：由 `AI-weather-models/Pangu/Main/draw-pangu-results.py` 等可视化模块读取 Output 数据后绘制生成

## Pangu 模型工作流

### 代码位置
所有 Pangu 核心代码都在 `AI-weather-models/Pangu/Main/` 目录：
- **get-data-pangu.py**：数据获取与预处理
  - 从 GFS、ERA5、ECMWF 等气象数据源下载数据
  - 将各种数据格式转换为 NPY 格式
  - 输出到 `Input/Pangu/`
  
- **run-pangu.py**：模型执行
  - 加载 `Models-weights/Pangu/pangu_weather_*.onnx` 
  - 读取 `Input/Pangu/` 的输入数据
  - 执行 ONNX 推理（支持 GPU 加速）
  - 输出到 `Output/Pangu/`
  
- **draw-pangu-results.py**：可视化绘图
  - 读取 `Output/Pangu/` 的预报数据
  - 使用 Cartopy 生成专业气象图
  - 输出到 `Run-output-png/Pangu/`
  
- **main-pangu-workflow.py**：完整工作流
  - 集成上述三个模块
  - 按顺序执行：数据获取 → 模型推理 → 结果可视化
  - 支持命令行参数配置

### 数据流向图
```
外部数据源（GFS、ERA5、ECMWF API）
         ↓
AI-weather-models/Pangu/Main/get-data-pangu.py (下载原始文件)
         ↓
Input/Pangu_raw/  (原始NC/GRIB文件存储)
         ↓
AI-weather-models/Pangu/Main/get-data-pangu.py (转换处理)
         ↓
Input/Pangu/  (处理后的NPY数据)
         ↓
AI-weather-models/Pangu/Main/run-pangu.py  (+ Models-weights/Pangu/)
         ↓
Output/Pangu/  (预报输出)
         ↓
AI-weather-models/Pangu/Main/draw-pangu-results.py
         ↓
Run-output-png/Pangu/  (可视化结果)
```

**工作流说明**：
1. 数据下载阶段：从外部 API 下载原始数据 → 保存到 `Input/*_raw/`
2. 数据处理阶段：读取 `Input/*_raw/` 中的原始文件，转换格式 → 保存到 `Input/*/`
3. 模型推理阶段：读取处理后的数据 `Input/*/` + 模型权重 → 执行推理 → 输出到 `Output/*/`
4. 结果可视化阶段：读取 `Output/*/` 的预报结果 → 绘制图片 → 保存到 `Run-output-png/*/`

## Fuxi 和 AIFS 模型说明

### Fuxi 模型
**代码位置**：`AI-weather-models/Fuxi/Main/` 和 `AI-weather-models/Fuxi/FuXi-main/`
- 支持多个预报时效（short、medium、long）
- 相应模型权重在 `Models-weights/Fuxi/`
- 输入数据在 `Input/Fuxi/`
- 输出数据在 `Output/Fuxi/`
- 可视化输出在 `Run-output-png/Fuxi/`

### AIFS 模型
**代码位置**：`AI-weather-models/AIFS/draw-aifs-from-open-store/`
- 主要脚本：`draw_aifs_snow.py`（AIFS 降雪可视化）
- 输入数据：`AI-weather-models/aifs-open-data/`（开源 GRIB2 数据）
- 可视化输出：`AI-weather-models/aifs-open-data-output-png/`

## 快速开始

### 完整工作流（推荐）
```bash
cd AI-weather-models/Pangu/Main/
python main-pangu-workflow.py \
    --data-source GFS \
    --init-datetime 2025121200 \
    --model-type 24 \
    --run-times 8
```

### 独立运行各步骤
```bash
# 第1步：获取并转换数据到 Input/Pangu/
cd AI-weather-models/Pangu/Main/
python get-data-pangu.py

# 第2步：运行模型推理，输出到 Output/Pangu/
python run-pangu.py

# 第3步：绘制可视化图片到 Run-output-png/Pangu/
python draw-pangu-results.py
```

## 文件路径说明

所有代码文件使用**相对路径**访问数据目录：

| 模块 | 执行位置 | 回溯层级 | 数据目录相对路径 |
|-----|--------|--------|------------------|
| Pangu 数据获取 | `AI-weather-models/Pangu/Main/` | ../../../ | 原始数据：`../../../Input/Pangu_raw/` |
| Pangu 数据获取 | `AI-weather-models/Pangu/Main/` | ../../../ | 处理数据：`../../../Input/Pangu/` |
| Pangu 模型执行 | `AI-weather-models/Pangu/Main/` | ../../../ | 输入：`../../../Input/Pangu/` |
| Pangu 模型执行 | `AI-weather-models/Pangu/Main/` | ../../../ | 输出：`../../../Output/Pangu/` |
| Pangu 可视化 | `AI-weather-models/Pangu/Main/` | ../../../ | 输入：`../../../Output/Pangu/` |
| Pangu 可视化 | `AI-weather-models/Pangu/Main/` | ../../../ | 输出：`../../../Run-output-png/Pangu/` |
| 模型权重加载 | `AI-weather-models/Pangu/Main/` | ../../../ | 模型：`../../../Models-weights/Pangu/` |

---

# AI-weather-models
A repository for AI weather models run, research and application


# 1.0

## 1.1.0
- inherit from old repository
- reconstruct the structure
- add README.md
- add LICENSE
- contain Fuxi, Pangu, AIFS 1.0
- Fuxi has not been reconstructed
- now add AIFS-ENS 1.0

- aim to build a unified repository for AI weather models

