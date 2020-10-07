import numpy as np
from devito import *

import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

from memory_profiler import profile
import matplotlib.pyplot as plt

configuration['log-level'] = 'WARNING'
torch.set_num_threads(8)
torch.set_default_tensor_type('torch.FloatTensor')


@profile
def conv(nx, ny, nch, n, m):
    # Image size
    dt = np.float32
    px, py = n//2 + 1, m//2 + 1
    x, y, c = SpaceDimension("x"), SpaceDimension("y"), Dimension("c")
    grid = Grid((nch, nx+2*px, ny+2*py), dtype=dt, dimensions=(c, x, y))

    stride = 2

    # Image
    im_in = Function(name="imi", grid=grid, space_order=1)
    input_data = np.zeros([nch, nx+2*px, ny+2*py])
    input_data[:, px:-px, py:-py] = np.linspace(-1, 1, nx*ny*nch).reshape(nch, nx, ny)
    im_in.data[:] = input_data.astype(np.float32)

    # Output
    im_out = Function(name="imo", grid=grid, space_order=1)
    im_out.data

    # Weights
    i, j = Dimension("i"), Dimension("j")
    W = Function(name="W", dimensions=(c, i, j), shape=(nch, n, m), grid=grid)
    # popuate weights with deterministic values
    for i in range(nch):
        W.data[i, :, :] = np.linspace(i, i+(n*m), n*m).reshape(n, m)

    # Convlution
    conv = sum([W[c, i2, i1]*im_in[c, x+i1-n//2, y+i2-m//2]
                for i1 in range(n) for i2 in range(m)])

    op = Operator(Eq(im_out, conv))
    op()

    # then return im_our.data[::stride, ::stride] .... if stride, and batchsize jut another dim like 6/7
    return im_out.data[:, px:-px, py:-py]


def conv_torch(nx, ny, nch, n, m):

    with profiler.profile(profile_memory=True, record_shapes=True) as prof:
        with torch.no_grad():

            convt = nn.Conv2d(nch, nch, (n, m), stride=(1, 1), padding=(n//2, m//2),
                              bias=False)

            ww = np.zeros((nch, nch, n, m), dtype=np.float32)
            for i in range(nch):
                ww[i, i, :, :] = np.linspace(i, i+(n*m), n*m).reshape(n, m).T

            convt.weight[:] = torch.from_numpy(ww)

            input_data = np.linspace(-1, 1, nx*ny*nch).reshape(1, nch, nx, ny)
            im_in = torch.from_numpy(input_data.astype(np.float32))

            im_out = convt(im_in)

    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    return im_out.numpy()[0, ...]


if __name__ == '__main__':
    nx, ny, nch = 1024, 1024, 4
    n, m = 3, 3
    res1 = conv(nx, ny, nch, n, m)
    res2 = conv_torch(nx, ny, nch, n, m)

    err = np.linalg.norm(res1 - res2)/np.linalg.norm(res1)
    plt.imshow(res1.sum(axis=0) - res2.sum(axis=0), cmap="seismic",
               vmin=-1e-1, vmax=1e-1)
    plt.colorbar()
    plt.show()

    print("Difference between devito and pytorch is %2.2e \n" % err)
