import numpy as np

# Specify the path to the .npy file
path1 = r'input\testinput_surface.npy'
path2 = r'C:\Users\lyz13\OneDrive\Desktop\AI_models_research\Pangu-Weather-Release\output\output_surface_2021041700+06h.npy'

# Load the .npy file using numpy
data1 = np.load(path1)
data2 = np.load(path2)

# Access and manipulate the data as needed
# For example, you can print the shape of the data
print(data1.shape)
print(data2.shape)

para = 0  # msl, 10u, 10v, 2t
lat = 360
for i in range(4):
    print(data1[i, lat])
    print(data2[i, lat])

# Find and print the minimum pressure value
min_pressure1 = np.min(data1[0])  # Assuming the first parameter is pressure
min_pressure2 = np.min(data2[0])  # Assuming the first parameter is pressure

print(f"Minimum pressure in data1: {min_pressure1}")
print(f"Minimum pressure in data2: {min_pressure2}")