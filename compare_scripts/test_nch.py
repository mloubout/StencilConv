import numpy as np
from devito import *

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

from memory_profiler import profile

configuration['log-level'] = 'WARNING'
torch.set_num_threads(8)
torch.set_default_tensor_type('torch.FloatTensor')


@profile
def conv(nx, ny, nchi, ncho, n, m, ngroup=1):
    # Image size
    dt = np.float32
    x, y, ci, co = SpaceDimension("x"), SpaceDimension("y"), Dimension("ci"), Dimension("co")
    grid = Grid((nx, ny), dtype=dt, dimensions=(x, y))

    stride = 2

    # Image
    im_in = Function(name="imi", dimensions=(ci, x, y),
                     shape=(nchi, nx, ny), grid=grid, space_order=n//2)
    input_data = np.linspace(-1, 1, nx*ny*nchi).reshape(nchi, nx, ny)
    im_in.data[:] = input_data.astype(np.float32)

    # Output
    im_out = Function(name="imo", dimensions=(co, x, y),
                      shape=(ncho, nx, ny), grid=grid, space_order=m//2)
    im_out.data

    # Weights
    i, j = Dimension("i"), Dimension("j")
    W = Function(name="W", dimensions=(co, ci, i, j), shape=(ncho, nchi, n, m), grid=grid, space_order=0)
    # popuate weights with deterministic values
    for ii in range(ncho):
        for jj in range(nchi):
            W.data[ii, jj, :, :] = np.linspace(ii+jj, ii+jj+(n*m), n*m).reshape(n, m)

    # Convlution
    conv = [Eq(im_out, im_out + sum([W[co, ci, i1, i2] * im_in[ci, x+i1-n//2, y+i2-m//2]
                                       for i1 in range(n) for i2 in range(m)]))]
    op = Operator(conv)
    op()

    return im_out.data, im_in.data


def grad(nx, ny, nchi, ncho, n, m, xdat, dydat):
    # Image size
    dt = np.float32
    x, y, ci, co = SpaceDimension("x"), SpaceDimension("y"), Dimension("ci"), Dimension("co")
    grid = Grid((nx, ny), dtype=dt, dimensions=(x, y))

    stride = 2

    # Image
    X = Function(name="xin", dimensions=(ci, x, y),
                 shape=(nchi, nx, ny), grid=grid, space_order=n//2)

    # Output
    dy = Function(name="dy", dimensions=(co, x, y),
                  shape=(ncho, nx, ny), grid=grid, space_order=n//2)

    # Weights
    i, j = Dimension("i"), Dimension("j")
    dW = Function(name="dW", dimensions=(co, ci, i, j), shape=(ncho, nchi, n, m), grid=grid)

    # Convolution
    grad_eq = Inc(dW[co, ci, i, j], dy[co, x, y]*X[ci, x+i-n//2, y+j-m//2])
    op = Operator(grad_eq)
    op.cfunction

    X.data[:] = xdat[:]
    dy.data[:] = dydat[:]

    op()

    return dW.data[:]


def conv_torch(nx, ny, nchi, ncho, n, m):

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        convt = nn.Conv2d(nchi, ncho, (n, m), stride=(1, 1),
                          padding=(n//2, m//2), bias=False)

        ww = np.zeros((ncho, nchi, n, m), dtype=np.float32)
        for i in range(ncho):
            for j in range(nchi):
                ww[i, j, :, :] = np.linspace(i+j, i+j+(n*m), n*m).reshape(n, m)

        convt.weight[:] = torch.from_numpy(ww)

        input_data = np.linspace(-1, 1, nx*ny*nchi).reshape(1, nchi, nx, ny)
        im_in = torch.from_numpy(input_data.astype(np.float32))

        im_out = convt(im_in)

        loss = .5*torch.norm(im_out)**2
        grad_t = torch.autograd.grad(loss, convt.parameters())

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    return im_out, grad_t


if __name__ == '__main__':
    nx, ny, nchi, ncho = 128, 128, 4,4 
    n, m = 3, 3
    res1, x = conv(nx, ny, nchi, ncho, n, m)
    res2, grad_t = conv_torch(nx, ny, nchi, ncho, n, m)
    err = np.linalg.norm(res1 - res2.detach().numpy())/np.linalg.norm(res1)
    print("Difference between devito and pytorch is %2.2e \n" % err)

    grad_d = grad(nx, ny, nchi, ncho, n, m, x, res1)
    err = np.linalg.norm(grad_d - grad_t[0].detach().numpy())/np.linalg.norm(grad_d)
    print("Gradient difference between devito and pytorch is %2.2e \n" % err)
