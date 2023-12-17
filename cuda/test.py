from numba import cuda
import numpy as np

# CUDA kernel
@cuda.jit
def add_kernel(x, y, out):
    tx = cuda.threadIdx.x  # this is the unique thread ID within a 1D block
    ty = cuda.blockIdx.x   # Similarly, this is the unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid
    
    start = tx + ty * block_size
    stride = block_size * grid_size

    # assuming x and y inputs are same length
    for i in range(start, x.shape[0], stride):
        out[i] = x[i] + y[i]

# Host code   
n = 100000
x = np.arange(n).astype(np.float32)
y = 2 * x
out = np.empty_like(x)

# Copy the arrays to the device
x_device = cuda.to_device(x)
y_device = cuda.to_device(y)
out_device = cuda.device_array_like(x)

# Configure the blocks
threadsperblock = 128
blockspergrid = (x.size + (threadsperblock - 1)) // threadsperblock

# Start the kernel 
add_kernelblockspergrid, threadsperblock

# Copy the result back to the host
out_device.copy_to_host(out)

print(out[:10])  # print the first 10 elements of the result
