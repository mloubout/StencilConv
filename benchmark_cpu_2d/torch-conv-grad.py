import sys
import numpy as np
import torch
import time
torch.set_num_threads(8)
torch.set_default_tensor_type('torch.FloatTensor')


def conv(nx, ny, nchi, ncho, n, m):


    # Define convolution operator
    convt = torch.nn.Conv2d(nchi, ncho, (n, m), stride=(1, 1),
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

    # Popuate weights with deterministic values
    convt.weight[:] = torch.from_numpy(ww)

    # First is compilation from the look of it since first call always
    # way slower
    im_out = convt(im_in)

    return convt, im_in, im_out


if __name__ == '__main__':

    k = int(sys.argv[1])
    n = 2**int(sys.argv[2])
    nch = 2**int(sys.argv[3])

    t0 = time.time()
    op, im_in, im_out = conv(n, n, nch, nch, k, k)
    loss = .5*torch.norm(im_out)**2
    op_build_time = time.time() - t0

    t0 = time.time()
    for j in range(50):
        torch.autograd.grad(loss, op.parameters(), retain_graph=True)
    op_run_time = time.time() - t0

    print(sys.argv[3], sys.argv[1], sys.argv[2], op_build_time,
          op_run_time, -1)
