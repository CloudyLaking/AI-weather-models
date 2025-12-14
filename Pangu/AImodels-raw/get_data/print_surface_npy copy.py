import numpy as np

# Specify the path to the .npy file
file_path = r'input\gfs_2015011500.npy'

# Load the .npy file using numpy
data = np.load(file_path)

# Access and manipulate the data as needed
# For example, you can print the shape of the data
print(data.shape)
para=3#msl, 10u, 10v, 2t
lat=360
print(data[para,lat])