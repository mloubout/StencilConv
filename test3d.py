import numpy as np
from devito import *


# Image size
dt = np.float32
nx, ny, nz, nch = 256, 256, 128, 4
x, y, z, c = SpaceDimension("x"), SpaceDimension("y"), SpaceDimension("z"), SpaceDimension("c")
grid = Grid((nx, ny, nz, nch), dimensions=(x,y,z,c), dtype=dt)

stride = 2

# Image
im_in = Function(name="imi", grid=grid, space_order=1)
im_in.data[:, :] = np.linspace(-1, 1, nx*ny*nz).reshape(nx, ny, nz)

# Output
im_out = Function(name="imo", grid=grid, space_order=1)

# Weights
i, j, k = Dimension("i"), Dimension("j"), Dimension("k")
l, m, n = 3, 3, 3
W = Function(name="W", dimensions=(i, j, k, c), shape=(l, m, n, nch), grid=grid)
# popuate weights with deterministic values
for i in range(nch):
    W.data[:, :, :, i] = np.linspace(i, i+(l*m*n), l*m*n).reshape(l, m, n)

# Convlution
conv = sum([W[i1, i2, i3, c]*im_in[x+i1-l//2, y+i2-m//2, z+i3-n//2, c]
            for i1 in range(l) for i2 in range(m) for i3 in range(n)])

op = Operator(Eq(im_out, conv))
op()

# then return im_our.data[::stride, ::stride] .... if stride, and batchsize jut another dim like 6/7

print(norm(im_out))