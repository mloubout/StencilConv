import numpy as np
from devito import *

import torch
import torch.nn as nn

def conv(nx, ny, nz, nchi, ncho, l, m, n):
    # Image size
    dt = np.float32
    x, y, z, ci, co = (SpaceDimension("x"), SpaceDimension("y"), SpaceDimension("z"),
                       Dimension("ci"), Dimension("co"))
    grid = Grid((nx, ny, nz), dimensions=(x,y,z), dtype=dt)

    stride = 2

    # Image
    im_in = Function(name="imi", dimensions=(ci, x, y, z),
                     shape=(nchi, nx, ny, nz), grid=grid, space_order=n//2)
    im_in.data[:] = np.linspace(-1, 1, nx*ny*nz*nchi).reshape(nchi, nx, ny, nz)

    # Output
    im_out = Function(name="imo", dimensions=(co, x, y, z),
                      shape=(ncho, nx, ny, nz), grid=grid, space_order=m//2)
    im_out.data

    # Weights
    i, j, k = Dimension("i"), Dimension("j"), Dimension("k")
    W = Function(name="W", dimensions=(co, ci, i, j, k), shape=(ncho, nchi, l, m, n), grid=grid, space_order=0)
    # popuate weights with deterministic values
    for ii in range(ncho):
        for jj in range(nchi):
            W.data[ii, jj, :, :, :] = np.linspace(ii+jj, ii+jj+(l*m*n), l*m*n).reshape(l, m, n)

    # Convlution
    conv = Eq(im_out, im_out + sum([W[co, ci, i1, i2, i3] * im_in[ci, x+i1-l//2, y+i2-m//2, z+i3-n//2]
                                   for i1 in range(l) for i2 in range(m) for i3 in range(n)]))
    op = Operator(conv)
    op()

    # then return im_our.data[::stride, ::stride] .... if stride, and batchsize just another dim like 6/7
    print(norm(im_out))
    return im_out.data, im_in.data

def conv_torch(nx, ny, nz, nchi, ncho, l, m, n):
    with torch.no_grad():

        convt = nn.Conv3d(nchi, ncho, (l, m, n), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)

        ww = np.zeros((nchi, ncho, l, m, n), dtype=np.float32)
        for ii in range(ncho):
            for jj in range(nchi):
                ww[ii, jj, :, :, :] = np.linspace(ii+jj, ii+jj+(l*m*n), l*m*n).reshape(l, m, n)

        convt.weight[:] = torch.from_numpy(ww)

        in_array = np.linspace(-1, 1, nx*ny*nz*nchi).reshape(1, nchi, nx, ny, nz).astype(np.float32)
        im_in = torch.from_numpy(in_array)
        im_out = convt(im_in)
        print(np.linalg.norm(im_out.detach().numpy()))
        return im_out


if __name__ == '__main__':
    nx, ny, nz, nchi, ncho = 32, 16, 16, 4, 4
    l, m, n = 3, 3, 3
    res1, x = conv(nx, ny, nz, nchi, ncho, l, m, n)
    res2 = conv_torch(nx, ny, nz, nchi, ncho, l, m, n)
    err = np.linalg.norm(res1 - res2.detach().numpy())/np.linalg.norm(res1)
    print("Difference between devito and pytorch is %2.2e \n" % err)
