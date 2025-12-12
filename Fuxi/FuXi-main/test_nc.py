import xarray as xr
import sys

def main():
    nc_file = input("请输入要打开的 NetCDF 文件路径: ")
    try:
        ds = xr.open_dataset(nc_file, engine="netcdf4")
    except Exception as e:
        print(f"无法打开文件 {nc_file}: {e}")
        sys.exit(1)
    
    print("文件尺寸信息:")
    print(ds.dims)
    
    print("\n变量列表及其维度:")
    for var in ds.data_vars:
        print(f"{var}，shape: {ds[var].shape}")
    
    print("\n所有变量详细信息:")
    print(ds)

    # 打印部分变量样本，并检查空值
    print("\n变量样本及空值检查:")
    for var, da in ds.data_vars.items():
        print(f"\n变量: {var}")
        # 打印前五个数据样本（展开成一维）
        sample = da.values[:500]
        print("前5个样本:", sample)
        # 计算空值个数
        missing_count = int(da.isnull().sum().values)
        print("空值数量:", missing_count)
    
    ds.close()

if __name__ == "__main__":
    main()