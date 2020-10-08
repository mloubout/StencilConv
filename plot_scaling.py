import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
font = {'family': 'serif',
        'size': 8}
matplotlib.rc('font', **font)

def img_size(n_points):
    return [2**(5 + j) for j in range(n_points)]

filenames = ['scaling-devito.txt',
             'scaling-torch.txt',
             'scaling-torch-gpu.txt']

run_times = {}
for file in filenames:
    run_times[file] = {"3": [], "5": [], "7": [], "11": []}
2
for file in filenames:
    with open(file) as f:
        for content in f.readlines():
            n, run_time = content.rstrip().split(',')
            run_times[file][str(n)].append(float(run_time))

colors = [(0.0,0.0,0.0),
          (0.0,0.584,1.0),
          (1.0,0.0,0.286),
          (0.0,0.584,0.239),
          '#8a8a8a',
          '#a1c0ff',
          '#ff9191',
          '#91eda2',
          '#8a8a8a',
          '#a1c0ff',
          '#ff9191',
          '#91eda2']

fig = plt.figure("Scaling devito", figsize=(8, 3))

counter = 0
for file in filenames:
    for n_str in run_times[file].keys():
        n = float(n_str)

        if file == filenames[-1]:
            linestyle = '--'
        else:
            linestyle = '-'
        plt.plot(img_size(len(run_times[file][n_str])),
                 run_times[file][n_str],
                 color=colors[counter], linewidth=0.7, linestyle=linestyle,
                 label=file[8:-4] + " — " + r"$k={}$".format(int(n)))
        plt.scatter(img_size(len(run_times[file][n_str])),
                    run_times[file][n_str],
                    color=colors[counter], s=0.8)

        counter += 1

plt.legend(fontsize=6, ncol=3)
plt.ylabel("wall-clock time (s)", fontsize=8)
plt.xlabel(r"$n$", fontsize=10)
plt.title("50 calls to a " + r"$k \times k \ conv$" + " — image size: "
          + r"$n \times n$")
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=.2)
plt.savefig('scaling.png' ,format='png', bbox_inches='tight',
            dpi=400, pad_inches=.05)
plt.close(fig)
