from ecmwfapi import ECMWFDataServer
import os

# 创建输出目录
output_dir = 'raw_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 指定输出文件路径
output_file = os.path.join(output_dir, 'output.grib')

server = ECMWFDataServer()

server.retrieve({
    "class": "ti",
    "dataset": "tigge",
    "date": "2013-11-07",
    "expver": "prod",
    "grid": "0.5/0.5",
    "levtype": "sfc",
    "origin": "ecmf",
    "param": "151/165/166/167",
    "step": "0",
    "time": "00:00:00/12:00:00",
    "type": "fc",
    "target": output_file
})