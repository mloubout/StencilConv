from memory_profiler import profile
import time
import torch
import numpy as np
from devito import (Grid, Eq, SpaceDimension, Function, Dimension, Operator,
                    configuration)
configuration['log-level'] = 'WARNING'
torch.set_num_threads(8)
torch.set_default_tensor_type('torch.FloatTensor')


@profile
def conv(nx, ny, nch, n, m):

    start_time = time.time()

    # Image size
    dt = np.float32
    x, y, c = SpaceDimension("x"), SpaceDimension("y"), SpaceDimension("c")
    grid = Grid((nch, nx, ny), dtype=dt, dimensions=(c, x, y))

    stride = 2

    # Image
    im_in = Function(name="imi", grid=grid, space_order=1)
    im_in.data[:, :, :] = np.linspace(-1, 1, nx*ny*nch).reshape(nch, nx, ny)

    # Output
    im_out = Function(name="imo", grid=grid, space_order=1)
    im_out.data

    # Weights
    i, j = Dimension("i"), Dimension("j")
    W = Function(name="W", dimensions=(i, j, c), shape=(n, m, nch), grid=grid)
    # popuate weights with deterministic values
    for i in range(nch):
        W.data[:, :, i] = np.linspace(i, i+(n*m), n*m).reshape(n, m)

    # Convlution
    conv = sum([W[i2, i1, c]*im_in[c, x+i1-n//2, y+i2-m//2]
                for i1 in range(n) for i2 in range(m)])

    op = Operator(Eq(im_out, conv))

    run_conv(op)
    print(('Devito elapsed time: %4.4f') % (time.time() - start_time))

    # then return im_our.data[::stride, ::stride] .... if stride, and batchsize jut another dim like 6/7

    return im_out.data


@profile
def conv_torch(nx, ny, nch, n, m):
    start_time = time.time()

    convt = torch.nn.Conv2d(nch, nch, (n, m), stride=(1, 1), padding=(1, 1),
                            bias=False)

    ww = np.zeros((nch, nch, n, m), dtype=np.float32)
    for i in range(nch):
        ww[i, i, :, :] = np.linspace(i, i+(n*m), n*m).reshape(n, m).T

    convt.weight[:] = torch.from_numpy(ww)

    in_array = np.linspace(-1, 1, nx*ny*nch).reshape(1, nch, nx, ny)
    im_in = torch.from_numpy(in_array.astype(np.float32))

    run_conv_torch(convt, im_in)
    print(('PyTorch elapsed time: %4.4f') % (time.time() - start_time))


@profile
def run_conv(op):
    for j in range(100):
        op()


@profile
def run_conv_torch(convt, im_in):
    for j in range(100):
        with torch.no_grad():
            convt(im_in)


if __name__ == '__main__':
    nx, ny, nch = 2048, 2048, 4
    n, m = 3, 3
    conv(nx, ny, nch, n, m)
    conv_torch(nx, ny, nch, n, m)
