import numpy as np
import torch
import torch.nn as nn
import time
torch.set_num_threads(8)
torch.set_default_tensor_type('torch.FloatTensor')
device = torch.device('cuda')

def conv_torch(nx, ny, nch, n, m, n_runs):

    start_time = time.time()

    with torch.no_grad():

        convt = nn.Conv2d(nch, nch, (n, m), stride=(1, 1), padding=(1, 1),
                          bias=False)

        ww = np.zeros((nch, nch, n, m), dtype=np.float32)
        for i in range(nch):
            ww[i, i, :, :] = np.linspace(i, i+(n*m), n*m).reshape(n, m).T

        convt.weight[:] = torch.from_numpy(ww)

        convt = convt.to(device)

        input_data = np.linspace(-1, 1, nx*ny*nch).reshape(1, nch, nx, ny)
        im_in = torch.from_numpy(input_data.astype(np.float32))
        im_in = im_in.to(device)

        for j in range(n_runs):
            im_out = convt(im_in)

    return time.time() - start_time


if __name__ == '__main__':
    nch = 2
    n, m = 3, 3

    nx_list = [2**j for j in range(5, 15)]
    run_times = []
    try:
        for nx in nx_list:
            run_times.append(conv_torch(nx, nx, nch, n, m, 200))
    except:
        print("not enough memory")

    with open('scaling-torch-gpu-conv-run-times.txt', 'w') as f:
        for item in run_times:
            f.write("%s\n" % item)
