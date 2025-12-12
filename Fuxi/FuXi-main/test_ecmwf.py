#!/usr/bin/env python
import os
from datetime import datetime
from ecmwfapi import ECMWFDataServer

def main():
    # 设定一个测试用的起始时刻，请根据实际情况修改
    test_start_time = datetime(2024, 9, 27, 0, 0, 0)
    
    req_params = {
        "class": "ti",
        "dataset": "tigge",
        "date": test_start_time.strftime('%Y-%m-%d'),
        "time": test_start_time.strftime('%H:00:00'),
        "expver": "prod",
        "grid": "0.5/0.5",
        "levtype": "sfc",
        "origin": "ecmf",
        "param": "151",
        "step": "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168",
        "type": "fc",
        "target": os.path.join("raw_input", "ecmwf_forecast.grib")
    }
    
    os.makedirs("raw_input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("开始下载 ECMWF 预报数据...")
    ECMWFDataServer().retrieve(req_params)
    print("ECMWF预报数据下载完成。")

if __name__ == "__main__":
    main()