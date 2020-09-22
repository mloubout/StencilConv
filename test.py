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
def conv(nx, ny, nch, n, m):
    # Image size
    dt = np.float32
    x, y, c = SpaceDimension("x"), SpaceDimension("y"), Dimension("c")
    grid = Grid((nx, ny, nch), dtype=dt, dimensions=(x, y, c))

    stride = 2

    # Image
    im_in = Function(name="imi", grid=grid, space_order=1)
    input_data = np.linspace(-1, 1, nx*ny*nch).reshape(nch, nx, ny)
    im_in.data[:] = input_data.transpose(1, 2, 0).astype(np.float32)

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
    conv = sum([W[i2, i1, c]*im_in[x+i1-n//2, y+i2-m//2, c]
                for i1 in range(n) for i2 in range(m)])

    op = Operator(Eq(im_out, conv))
    op()

    # then return im_our.data[::stride, ::stride] .... if stride, and batchsize jut another dim like 6/7
    return im_out.data


def conv_torch(nx, ny, nch, n, m):

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with torch.no_grad():

            convt = nn.Conv2d(nch, nch, (n, m), stride=(1, 1), padding=(1, 1),
                              bias=False)

            ww = np.zeros((nch, nch, n, m), dtype=np.float32)
            for i in range(nch):
                ww[i, i, :, :] = np.linspace(i, i+(n*m), n*m).reshape(n, m).T

            convt.weight[:] = torch.from_numpy(ww)

            input_data = np.linspace(-1, 1, nx*ny*nch).reshape(1, nch, nx, ny)
            im_in = torch.from_numpy(input_data.astype(np.float32))

            im_out = convt(im_in)

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    return im_out.numpy().transpose(0, 2, 3, 1)


if __name__ == '__main__':
    nx, ny, nch = 1024, 1024, 4
    n, m = 3, 3
    res1 = conv(nx, ny, nch, n, m)
    res2 = conv_torch(nx, ny, nch, n, m)
    err = np.linalg.norm(res1 - res2)/np.linalg.norm(res1)
    print("Difference between devito and pytorch is %2.2e \n" % err)
