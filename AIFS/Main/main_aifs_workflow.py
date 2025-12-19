# AI-weather-models\AIFS\Main\main_aifs_workflow.py
# 整合工作流编排脚本 - 完整的数据获取、模型预报、绘图流程
# Uncomment the lines below to install the required packages

"""
pip install -q anemoi-inference[huggingface]==0.6.3 anemoi-models==0.5.0
pip install -q torch-geometric==2.4.0
pip install -q earthkit-regrid==0.4.0 ecmwf-opendata
pip install  flash_attn  (一般不行，需要手动下载精准对应系统的轮子安装，需查询相关教程)
pip install  cdsapi cartopy

请预先在系统用户目录中放好 cdsapi 的 .cdsapirc 文件与 ecmwf-opendata 的 .ecmwfapirc 文件
可运行环境示例 Python 3.12.3 
vGPU-48GB  (至少要32GB，最好直接上48GB/24*2GB)

"""

import os
import sys
from datetime import datetime, timedelta
import atexit

# 抑制临时文件清理错误(Windows特定)
def _suppress_tempfile_cleanup_errors():
    """抑制 earthkit 退出时的临时文件清理权限错误"""
    original_atexit_register = atexit.register
    def patched_atexit_register(func, *args, **kwargs):
        def wrapped(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except (PermissionError, OSError, NotADirectoryError):
                pass  # 忽略文件清理错误
        return original_atexit_register(wrapped, *args, **kwargs)
    atexit.register = patched_atexit_register

_suppress_tempfile_cleanup_errors()

# 导入本地模块
try:
    from get_data_aifs import AIFSDataDownloader, get_aifs_data
    from run_aifs import AIFSRunner, run_aifs_forecast
    from draw_aifs_results import AIFSResultDrawer, draw_aifs_results
except ImportError as e:
    print(f"[ERROR] Failed to import modules: {e}")
    print("[INFO] Make sure get_data_aifs.py, run_aifs.py, and draw_aifs_results.py are in the same directory")
    sys.exit(1)


def run_complete_aifs_workflow(init_datetime_str=None, lead_time=12,
                              device='cuda', draw_results=True,
                              skip_existing=True,
                              model_path=None,
                              use_huggingface=False,
                              input_dir='Input/AIFS',
                              raw_input_dir='Input/AIFS_raw',
                              output_dir='Output/AIFS',
                              image_dir='Run-output-png/AIFS',
                              data_source='ECMWF',
                              use_gpu_interp=True,
                              interp_res=0.5):
    """
    执行完整的 AIFS 天气预报工作流
    
    参数:
        init_datetime_str: 初始化时间字符串,格式 'YYYYMMDDHH',None则自动获取最新
        lead_time: 预报时效(小时)
        device: 计算设备,'cuda' 或 'cpu'
        draw_results: 是否绘制结果图
        skip_existing: 是否跳过已存在的文件
        model_path: 模型权重路径（为None则根据use_huggingface决定）
        use_huggingface: 是否使用 Hugging Face 模型（False=本地，True=HF）
        input_dir: 处理后输入数据目录
        raw_input_dir: 原始输入数据目录
        output_dir: 预报输出目录
        image_dir: 图像输出目录
        data_source: 数据源,'ECMWF'(最新数据)或 'ERA5'(历史数据)
        use_gpu_interp: 是否使用GPU加速插值(推荐,比CPU三角剖分快60-120倍)
        interp_res: 插值目标分辨率(度),0.5推荐,0.25更精细,1.0更快
        
    返回:
        dict: 工作流执行结果
    """
    
    # 自动定位项目根目录
    project_root = os.getcwd()
    if not os.path.exists(os.path.join(project_root, 'Input')):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        current = script_dir
        for _ in range(4):
            if os.path.exists(os.path.join(current, 'Input')):
                project_root = current
                break
            current = os.path.dirname(current)
            
    print(f"[INFO] Project root detected as: {project_root}")

    # 将相对路径转换为绝对路径
    if not os.path.isabs(input_dir):
        input_dir = os.path.normpath(os.path.join(project_root, input_dir))
    if not os.path.isabs(raw_input_dir):
        raw_input_dir = os.path.normpath(os.path.join(project_root, raw_input_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(project_root, output_dir))
    if not os.path.isabs(image_dir):
        image_dir = os.path.normpath(os.path.join(project_root, image_dir))
    
    print(f"[INFO] Using paths:")
    print(f"  Raw Input:       {raw_input_dir}")
    print(f"  Processed Input: {input_dir}")
    print(f"  Output:          {output_dir}")
    print(f"  Images:          {image_dir}\n")
    
    workflow_start = datetime.now()
    results = {
        'status': 'running',
        'init_datetime': init_datetime_str,
        'lead_time': lead_time,
        'start_time': workflow_start.strftime('%Y-%m-%d %H:%M:%S'),
        'data_files': [],
        'forecast_files': [],
        'image_files': [],
        'errors': [],
        'execution_time': None
    }
    
    print("\n" + "="*70)
    print("AIFS WEATHER FORECAST WORKFLOW")
    print("="*70)
    print(f"[START] Workflow initiated at {workflow_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =====================================================================
    # STEP 1: 数据获取
    # =====================================================================
    print("\n" + "-"*70)
    print(f"STEP 1: DATA ACQUISITION FROM {data_source.upper()}")
    print("-"*70)
    
    try:
        print(f"[INFO] Acquiring AIFS initial conditions from {data_source}...")
        
        # 创建数据下载器
        downloader = AIFSDataDownloader(
            input_dir=input_dir,
            raw_input_dir=raw_input_dir
        )
        
        # 根据数据源选择不同的处理方法
        if data_source.upper() == 'ERA5':
            if init_datetime_str is None:
                print("[ERROR] ERA5 requires explicit datetime (YYYYMMDDHH), cannot auto-get")
                results['status'] = 'failed'
                results['errors'].append('ERA5 requires explicit datetime')
                return results
            
            input_state = downloader.process_era5_data(
                datetime_str=init_datetime_str,
                skip_existing=skip_existing
            )
        else:  # ECMWF Open Data
            input_state = downloader.process_ecmwf_data(
                datetime_str=init_datetime_str,
                skip_existing=skip_existing
            )
        
        if input_state is None or 'date' not in input_state:
            print("[ERROR] Data acquisition failed")
            results['status'] = 'failed'
            results['errors'].append('Data acquisition failed')
            return results
        
        # 更新初始时间
        if init_datetime_str is None:
            init_datetime_str = input_state['date'].strftime('%Y%m%d%H')
            results['init_datetime'] = init_datetime_str
        
        # 统计字段数量
        num_fields = len(input_state.get('fields', {}))
        print(f"[OK] Data acquisition completed successfully")
        print(f"  Initial fields: {num_fields}")
        print(f"  Date: {input_state['date']}")
        
        # ===== 绘制初始场 =====
        if draw_results:
            print(f"\n[INFO] Plotting initial conditions...")
            try:
                init_state = input_state.copy()
                init_state['init_date'] = input_state['date']

                image_files = draw_aifs_results(
                    init_state,
                    init_datetime_str=init_datetime_str,
                    data_source=data_source,
                    use_gpu_interp=use_gpu_interp,
                    target_res=interp_res,
                    device=device
                )

                results['image_files'].extend(image_files)
                if image_files:
                    print(f"  [OK] Initial conditions plotted: {len(image_files)} images")

            except Exception as e:
                print(f"  [WARN] Failed to plot initial conditions: {e}")
        
    except Exception as e:
        print(f"[ERROR] Data acquisition error: {e}")
        results['status'] = 'failed'
        results['errors'].append(f'Data acquisition error: {e}')
        return results
    
    # =====================================================================
    # STEP 2: 模型预报
    # =====================================================================
    print("\n" + "-"*70)
    print("STEP 2: MODEL EXECUTION (AIFS Forecast)")
    print("-"*70)
    
    forecast_states = []
    output_files = []  # 初始化避免未绑定错误
    
    try:
        print(f"[INFO] Running AIFS forecast...")
        print(f"  Lead time: {lead_time} hours")
        print(f"  Device: {device}")
        
        # 创建执行器
        runner = AIFSRunner(
            device=device,
            model_path=model_path,
            output_dir=output_dir,
            use_huggingface=use_huggingface
        )
        
        if runner.runner is None:
            print("[ERROR] AIFS runner initialization failed")
            results['status'] = 'failed'
            results['errors'].append('AIFS runner initialization failed')
            return results
        
        # 执行预报
        output_files = runner.run_forecast(
            input_state=input_state,
            lead_time=lead_time,
            save_outputs=True,
            datetime_str=init_datetime_str,
            skip_existing=skip_existing
        )
        
        results['forecast_files'] = output_files
        
        if not output_files:
            print("[WARN] No new forecast outputs were generated (all existing)")
        else:
            print(f"[OK] Forecast execution completed successfully")
            print(f"  Generated outputs: {len(output_files)}")
        
        # 为了绘图,我们需要重新运行获取所有预报状态
        print(f"[INFO] Retrieving forecast states for visualization...")
        for state in runner.runner.run(input_state=input_state, lead_time=lead_time):
            forecast_states.append(state)
        
        print(f"[OK] Retrieved {len(forecast_states)} forecast states")
        
    except Exception as e:
        print(f"[ERROR] Model execution error: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['errors'].append(f'Model execution error: {e}')
        # 继续进行可视化(如果已有输出)
        if not output_files:
            return results
    
    # =====================================================================
    # STEP 3: 结果可视化
    # =====================================================================
    if draw_results and forecast_states:
        print("\n" + "-"*70)
        print("STEP 3: VISUALIZATION")
        print("-"*70)
        
        try:
            print(f"[INFO] Drawing forecast results...")
            
            total_images = 0
            
            # 为每个预报时次绘图
            for state_idx, state in enumerate(forecast_states, 1):
                print(f"\n  Generating images for state {state_idx}/{len(forecast_states)}...")
                
                # 添加初始日期信息用于绘图
                state['init_date'] = input_state['date']
                
                image_files = draw_aifs_results(
                    state,
                    init_datetime_str=init_datetime_str,
                    data_source=data_source,
                    use_gpu_interp=use_gpu_interp,
                    target_res=interp_res,
                    device=device
                )
                
                results['image_files'].extend(image_files)
                total_images += len(image_files)
            
            print(f"\n[OK] Visualization completed successfully")
            print(f"  Total images: {total_images}")
            
        except Exception as e:
            print(f"[ERROR] Visualization error: {e}")
            import traceback
            traceback.print_exc()
            results['errors'].append(f'Visualization error: {e}')
    
    # =====================================================================
    # 完成
    # =====================================================================
    workflow_end = datetime.now()
    execution_time = workflow_end - workflow_start
    
    results['status'] = 'completed'
    results['end_time'] = workflow_end.strftime('%Y-%m-%d %H:%M:%S')
    results['execution_time'] = str(execution_time)
    
    print("\n" + "="*70)
    print("WORKFLOW SUMMARY")
    print("="*70)
    print(f"[OK] Workflow completed at {workflow_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Execution time: {execution_time}")
    print(f"  Status: {results['status']}")
    print(f"  Initial conditions: {init_datetime_str}")
    print(f"  Forecast lead time: {lead_time} hours")
    print(f"  Forecast outputs: {len(results['forecast_files'])}")
    print(f"  Visualization images: {len(results['image_files'])}")
    
    if results['errors']:
        print(f"\n  Errors encountered:")
        for error in results['errors']:
            print(f"    - {error}")
    
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    # ===== 在这里设置运行参数 =====
    init_datetime = '2025121000'          # 起报时间,None 则自动获取最新
    lead_time = 24                         # 预报时效(小时),整6小时
    device = 'cuda'                       # 计算设备:'cuda' 或 'cpu'
    data_source = 'ERA5'                 # 数据源:'ECMWF' 或 'ERA5'
    draw_results = True                   # 是否绘制结果
    skip_existing = True                  # 是否跳过已存在文件
    use_gpu_interp = False                # 使用三角剖分(False推荐,稳定准确)
    interp_res = 0.5                      # 插值分辨率(度):仅use_gpu_interp=True时有效
    
    # ===== 模型源选择参数 =====
    use_huggingface = False               # False=使用本地模型, True=使用Hugging Face模型
    model_path = None                     # 本地模型路径(None则使用默认路径或HF)
    # ===== 参数设置完毕 =====
    
    results = run_complete_aifs_workflow(
        init_datetime_str=init_datetime,
        lead_time=lead_time,
        device=device,
        draw_results=draw_results,
        skip_existing=skip_existing,
        data_source=data_source,
        use_gpu_interp=use_gpu_interp,
        interp_res=interp_res,
        use_huggingface=use_huggingface,
        model_path=model_path
    )