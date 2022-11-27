import h5py
import numpy as np

arr = np.random.randn(1000)

with h5py.File('random.h5py', 'w') as f:
    dset = f.create_dataset("default", data=arr)
with h5py.File('random.h5py', 'r') as f:
    data = f['default']
    print(min(data))
    print(max(data))
    print(data[:15])
