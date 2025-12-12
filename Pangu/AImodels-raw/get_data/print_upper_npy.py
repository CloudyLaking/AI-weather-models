import numpy as np

# Specify the path to the .npy file
path1= r'input\testinput_upper.npy'
path2= r'input\input_upper.npy'
# Load the .npy file using numpy
data1 = np.load(path1)
data2 = np.load(path2)

# Access and manipulate the data as needed
# For example, you can print the shape of the data
print(data1.shape)
print(data2.shape)
para=0#z,q,t,u,v
h=0
lat=100
for i in range(5):
    print(data1[i,h,lat])
    print(data2[i,h,lat])