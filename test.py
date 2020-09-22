import numpy as np
from devito import *


# Image size
dt = np.float64
nx, ny, nch = 1024, 1024, 4
x, y, c = SpaceDimension("x"), SpaceDimension("y"), SpaceDimension("c")
grid = Grid((nx, ny, nch), dtype=dt, dimensions=(x, y, c))

stride = 2

# Image
im_in = Function(name="imi", grid=grid, space_order=1)
im_in.data[:, :] = np.linspace(-1, 1, nx*ny).reshape(nx, ny)

# Output
im_out = Function(name="imo", grid=grid, space_order=1)

# Weights
i, j = Dimension("i"), Dimension("j")
n, m = 3, 3
W = Function(name="W", dimensions=(i, j, c), shape=(n, m, nch), grid=grid)
# popuate weights with deterministic values
for i in range(nch):
    W.data[:, :, i] = np.linspace(i, i+(n*m), n*m).reshape(n, m)

# Convlution
conv = sum([W[i1, i2, c]*im_in[x+i1-n//2, y+i2-m//2, c]
            for i1 in range(n) for i2 in range(m)])

op = Operator(Eq(im_out, conv))
op()

# then return im_our.data[::stride, ::stride] .... if stride, and batchsize jut another dim like 6/7

print(norm(im_out))