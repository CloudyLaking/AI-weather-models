# AI-weather-models\AIFS\Main\main_aifs_workflow.py
# 整合工作流编排脚本 - 完整的数据获取、模型预报、绘图流程

import os
import sys
import argparse
from datetime import datetime, timedelta
import atexit

# 抑制临时文件清理错误（Windows特定）
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
                              input_dir='../../../Input/AIFS',
                              raw_input_dir='../../../Input/AIFS_raw',
                              output_dir='../../../Output/AIFS',
                              image_dir='../../../Run-output-png/AIFS',
                              fields_to_draw=None,
                              data_source='ECMWF'):
    """
    执行完整的 AIFS 天气预报工作流
    
    参数:
        init_datetime_str: 起报时间，格式 'YYYYMMDDHH'，为 None 则使用最新数据
        lead_time: 预报时效（小时）
        device: 计算设备 'cuda' 或 'cpu'
        draw_results: 是否绘制可视化结果
        skip_existing: 是否跳过已存在的数据和输出
        model_path: 本地模型文件路径（为 None 则使用默认或 Hugging Face）
        input_dir: 处理后输入数据目录
        raw_input_dir: 原始输入数据目录
        output_dir: 预报输出数据目录
        image_dir: 图片输出目录
        fields_to_draw: 要绘制的字段列表，为 None 则使用默认值
        data_source: 数据源，'ECMWF'（最新数据）或 'ERA5'（历史数据）
        
    返回:
        dict: 工作流执行结果
    """
    
    # 获取脚本所在目录，用于转换相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 将相对路径转换为绝对路径
    if not os.path.isabs(input_dir):
        input_dir = os.path.normpath(os.path.join(script_dir, input_dir))
    if not os.path.isabs(raw_input_dir):
        raw_input_dir = os.path.normpath(os.path.join(script_dir, raw_input_dir))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
    if not os.path.isabs(image_dir):
        image_dir = os.path.normpath(os.path.join(script_dir, image_dir))
    
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
                from draw_aifs_results import AIFSResultDrawer, draw_aifs_results
                
                # 创建初始状态用于绘图
                init_state = input_state.copy()
                init_state['init_date'] = input_state['date']
                
                # 绘制初始场
                image_files = draw_aifs_results(
                    init_state,
                    init_datetime_str=init_datetime_str,
                    fields_to_draw=fields_to_draw,
                    draw_wind=True
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
    
    try:
        print(f"[INFO] Running AIFS forecast...")
        print(f"  Lead time: {lead_time} hours")
        print(f"  Device: {device}")
        
        # 创建执行器
        runner = AIFSRunner(
            device=device,
            model_path=model_path,
            output_dir=output_dir
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
            # 仍继续进行可视化
        else:
            print(f"[OK] Forecast execution completed successfully")
            print(f"  Generated outputs: {len(output_files)}")
        
        # 为了绘图，我们需要重新运行获取所有预报状态
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
        # 继续进行可视化（如果已有输出）
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
            
            # 创建绘图器
            drawer = AIFSResultDrawer(output_dir=image_dir)
            
            # 默认绘制的字段（仅风场）
            if fields_to_draw is None:
                fields_to_draw = None  # draw_aifs_results 会自动绘制风场
            
            total_images = 0
            
            # 为每个预报时次绘图
            for state_idx, state in enumerate(forecast_states, 1):
                print(f"\n  Generating images for state {state_idx}/{len(forecast_states)}...")
                
                # 添加初始日期信息用于绘图
                state['init_date'] = input_state['date']
                
                image_files = draw_aifs_results(
                    state,
                    init_datetime_str=init_datetime_str,
                    fields_to_draw=fields_to_draw,
                    draw_wind=True
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
            # 不返回，继续完成工作流
    
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


def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='AIFS Weather Forecast Complete Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use latest ECMWF Open Data (default)
  python main_aifs_workflow.py
  
  # Specify initialization time with ECMWF
  python main_aifs_workflow.py --datetime 2025121200
  
  # Use ERA5 reanalysis data (requires datetime)
  python main_aifs_workflow.py --datetime 2024120100 --data-source ERA5
  
  # Use CPU instead of GPU
  python main_aifs_workflow.py --device cpu
  
  # Specify lead time and customize output fields
  python main_aifs_workflow.py --lead-time 24 --fields 2t 100u msl gh_500
        """
    )
    
    parser.add_argument(
        '--datetime',
        type=str,
        default=None,
        help="Initialization datetime in format 'YYYYMMDDHH' (default: use latest available)"
    )
    
    parser.add_argument(
        '--lead-time',
        type=int,
        default=12,
        help="Forecast lead time in hours (default: 12)"
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help="Computing device (default: cuda)"
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to local AIFS model checkpoint (default: Models-weights/AIFS/aifs-single-mse-1.0.ckpt or Hugging Face)"
    )
    
    parser.add_argument(
        '--skip-existing',
        type=bool,
        default=True,
        help="Skip existing data and outputs (default: True)"
    )
    
    parser.add_argument(
        '--no-draw',
        action='store_true',
        help="Skip visualization step"
    )
    
    parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        default=None,
        help="Fields to visualize (default: 2t 100u msl)"
    )
    
    parser.add_argument(
        '--data-source',
        type=str,
        default='ECMWF',
        choices=['ECMWF', 'ERA5'],
        help="Data source: ECMWF Open Data (latest) or ERA5 (historical, requires --datetime)"
    )
    
    args = parser.parse_args()
    
    # 执行工作流
    results = run_complete_aifs_workflow(
        init_datetime_str=args.datetime,
        lead_time=args.lead_time,
        device=args.device,
        model_path=args.model_path,
        draw_results=not args.no_draw,
        skip_existing=args.skip_existing,
        fields_to_draw=args.fields,
        data_source=args.data_source
    )
    
    # 根据结果返回退出码
    sys.exit(0 if results['status'] == 'completed' else 1)


if __name__ == '__main__':
    # 运行时携带命令行参数则使用CLI模式
    if len(sys.argv) > 1:
        main()
    else:
        # 直接使用预设参数运行（无需命令行参数）
        print("[INFO] Running with preset parameters")
        print("[INFO] Use --help to see available options\n")
        
        # ===== 在这里预设参数 =====
        init_datetime = '2025121400'          # 起报时间，None 则自动获取最新
        lead_time = 6                        # 预报时效（小时），整6小时
        device = 'cuda'                       # 计算设备：'cuda' 或 'cpu'
        data_source = 'ECMWF'                 # 数据源：'ECMWF' 或 'ERA5'
        draw_results = True                   # 是否绘制结果
        skip_existing = True                  # 是否跳过已存在文件
        fields_to_draw = None                 # 默认仅绘制风场
        # ===== 参数预设完毕 =====
        
        results = run_complete_aifs_workflow(
            init_datetime_str=init_datetime,
            lead_time=lead_time,
            device=device,
            draw_results=draw_results,
            skip_existing=skip_existing,
            fields_to_draw=fields_to_draw,
            data_source=data_source
        )
