import numpy as np
from devito import *

import torch
import torch.nn as nn


@profile
def conv(nx, ny, nz, nch, l, m, n):
    # Image size
    dt = np.float32
    x, y, z, c = SpaceDimension("x"), SpaceDimension("y"), SpaceDimension("z"), Dimension("c")
    grid = Grid((nch, nx, ny, nz), dimensions=(c,x,y,z), dtype=dt)

    stride = 2

    # Image
    im_in = Function(name="imi", grid=grid, space_order=1)
    im_in.data
    im_in.data[:] = np.linspace(-1, 1, nx*ny*nz*nch).reshape(nch, nx, ny, nz)

    # Output
    im_out = Function(name="imo", grid=grid, space_order=1)

    # Weights
    i, j, k = Dimension("i"), Dimension("j"), Dimension("k")
    W = Function(name="W", dimensions=(i, j, k, c), shape=(l, m, n, nch), grid=grid)
    # popuate weights with deterministic values
    for i in range(nch):
        W.data[:, :, :, i] = np.linspace(i, i+(l*m*n), l*m*n).reshape(l, m, n)

    # Convlution
    conv = sum([W[i1, i2, i3, c]*im_in[c, x+i1-l//2, y+i2-m//2, z+i3-n//2]
                for i1 in range(l) for i2 in range(m) for i3 in range(n)])

    op = Operator(Eq(im_out, conv))
    op()

    # then return im_our.data[::stride, ::stride] .... if stride, and batchsize just another dim like 6/7
    return im_out.data


@profile
def conv_torch(nx, ny, nz, nch, l, m, n):
    with torch.no_grad():

        convt = nn.Conv3d(nch, nch, (l, m, n), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        ww = np.zeros((nch, nch, l, m, n), dtype=np.float32)
        for i in range(nch):
            ww[i, i, :, :] = np.linspace(i, i+(l*m*n), l*m*n).reshape(l, m, n)

        convt.weight[:] = torch.from_numpy(ww)

        in_array = np.linspace(-1, 1, nx*ny*nz*nch).reshape(1, nch, nx, ny, nz).astype(np.float32)
        im_in = torch.from_numpy(in_array)
        im_out = convt(im_in)
        return im_out.detach().numpy()



if __name__ == '__main__':
    nx, ny, nz, nch = 256, 256, 128, 4
    l, m, n = 3, 3, 3
    res1 = conv(nx, ny, nz, nch, l, m, n)
    res2 = conv_torch(nx, ny, nz, nch, l, m, n)
    err = np.linalg.norm(res1 - res2)/np.linalg.norm(res1)
    print("Difference between devito and pytorch is %2.2e \n" % err)