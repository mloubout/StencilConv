import numpy as np
from devito import *
import time
configuration['log-level'] = 'WARNING'


def conv(nx, ny, nch, n, m, n_runs):

    start_time = time.time()

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
    for j in range(n_runs):
        op()

    return time.time() - start_time


if __name__ == '__main__':
    nx, ny, nch = 2048, 2048, 4
    n, m = 3, 3

    n_runs = [2**j for j in range(13)]
    run_times = []
    for i in n_runs:
        run_times.append(conv(nx, ny, nch, n, m, i))

    with open('devito-conv-run-times.txt', 'w') as f:
        for item in run_times:
            f.write("%s\n" % item)
