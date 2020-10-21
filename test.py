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
def conv(nx, ny, nchi, ncho, n, m):

    # Image size
    dt = np.float32
    x, y, ci, co = (SpaceDimension("x"), SpaceDimension("y"), Dimension("ci"),
                    Dimension("co"))
    grid = Grid((nchi, ncho, nx, ny), dtype=dt, dimensions=(ci, co, x, y))

    # Image
    im_in = Function(name="imi", dimensions=(ci, x, y),
                     shape=(nchi, nx, ny), grid=grid, space_order=n//2)
    input_data = np.linspace(-1, 1, nx*ny*nchi).reshape(nchi, nx, ny)
    im_in.data[:] = input_data.astype(np.float32)

    # Output
    im_out = Function(name="imo", dimensions=(co, x, y),
                      shape=(ncho, nx, ny), grid=grid, space_order=n//2)
    im_out.data

    # Weights
    i, j = Dimension("i"), Dimension("j")
    W = Function(name="W", dimensions=(co, ci, i, j), shape=(ncho, nchi, n, m),
                 grid=grid)

    # Popuate weights with deterministic values
    for i in range(ncho):
        for j in range(nchi):
            W.data[i, j, :, :] = np.linspace(i+j, i+j+(n*m), n*m).reshape(n, m)

    # Convlution
    conv = [Eq(im_out, im_out
               + sum([W[co, ci, i2, i1] * im_in[ci, x+i1-n//2, y+i2-m//2]
                      for i1 in range(n) for i2 in range(m)]))]

    op = Operator(conv)
    op.cfunction
    op()

    return im_out.data


def conv_torch(nx, ny, nchi, ncho, n, m):

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        convt = nn.Conv2d(nchi, ncho, (n, m), stride=(1, 1),
                          padding=(n//2, m//2), bias=False)

        ww = np.zeros((ncho, nchi, n, m), dtype=np.float32)
        for i in range(ncho):
            for j in range(nchi):
                ww[i, j, ...] = np.linspace(i + j,
                                            i + j + (n * m),
                                            n * m).reshape(n, m).T

        convt.weight[:] = torch.from_numpy(ww)

        input_data = np.linspace(-1, 1, nx*ny*nchi).reshape(1, nchi, nx, ny)
        im_in = torch.from_numpy(input_data.astype(np.float32))

        im_out = convt(im_in)

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    return im_out[0, ...].detach().numpy()


if __name__ == '__main__':

    nx, ny, nch = 1024, 1024, 4
    n, m = 11, 11

    res1 = conv(nx, ny, nch, nch, n, m)
    res2 = conv_torch(nx, ny, nch, nch, n, m)

    err = np.linalg.norm(res1 - res2)/np.linalg.norm(res1)
    print("Difference between devito and pytorch is %2.2e \n" % err)
