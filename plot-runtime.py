import numpy as np

import matplotlib
import matplotlib.pyplot as plt

sfmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
sfmt.set_powerlimits((0, 0))
font = {'family': 'serif',
        'size': 10}
matplotlib.rc('font', **font)

# 2048 \times 2048 \times 4
devito = [0.4786996841430664,
0.2679026126861572,
0.2818915843963623,
0.3137233257293701,
0.3860750198364258,
0.49344563484191895,
0.7353940010070801,
1.2463245391845703,
2.2015531063079834,
4.074259519577026,
7.90655517578125,
15.561073780059814,
30.949887990951538]

torch_cpu = [0.1500563621520996,
0.21469855308532715,
0.33217597007751465,
0.5664196014404297,
1.034461498260498,
2.5286500453948975,
3.986839532852173,
7.849538564682007,
15.588876962661743,
31.068026542663574,
62.09574627876282,
123.96818661689758,
247.32093930244446]

torch_gpu = [1.8867990970611572,
0.10767412185668945,
0.12624263763427734,
0.1653449535369873,
0.21707391738891602,
0.34631848335266113,
0.6060693264007568,
1.1219377517700195,
2.1559295654296875,
4.220719814300537,
16.538241147994995,
33.000428438186646,
66.2021119594574]

n_runs = [2**j for j in range(13)]

fig = plt.figure("Objective", dpi=200, figsize=(7, 2.5))
plt.plot(n_runs, devito, color='#be22d6', linewidth=1.0,
         label='devito on cpu')
plt.scatter(n_runs, devito, color='#be22d6', s=1.5)

plt.plot(n_runs, torch_cpu, color='#22c1d6', linewidth=1.0,
         label='torch on cpu')
plt.scatter(n_runs, torch_cpu, color='#22c1d6', s=1.5)

# plt.plot(n_runs, torch_gpu, color='#aab304', linewidth=1.0,
#          label='torch on gpu')
# plt.scatter(n_runs, torch_gpu, color='#aab304', s=1.5)
plt.title("image size: " + r"$2048 \times 2048 \times 4$")
plt.legend(fontsize=8)
plt.xlabel("number of calls to conv")
plt.ylabel("wall-clock time (s)")
plt.grid()
plt.savefig('run-time-2048x2048x4.png' ,format='png', bbox_inches='tight',
            dpi=200, pad_inches=.05)
plt.close(fig)


# 2048 \times 2048 \times 4
devito = [0.5416955947875977,
0.6014156341552734,
0.7627716064453125,
1.0925052165985107,
1.7689414024353027,
3.0511763095855713,
5.677820920944214,
10.932159662246704,
21.35869836807251,
42.18250894546509,
83.92631959915161,
167.43272614479065,
334.60024785995483]

torch_cpu = [0.22685670852661133,
0.317471981048584,
0.4452023506164551,
0.7038383483886719,
1.2255847454071045,
2.2661678791046143,
4.351020812988281,
8.514991521835327,
16.81299924850464,
33.584160804748535,
66.8525083065033,
133.59986019134521,
266.8670337200165]

torch_gpu = [1.9679501056671143,
0.20565438270568848,
0.23413658142089844,
0.2850010395050049,
0.37809228897094727,
0.5705912113189697,
0.9654874801635742,
1.7543504238128662,
3.3311259746551514,
6.479235887527466,
25.312326908111572,
50.53777718544006,
103.28148031234741]

n_runs = [2**j for j in range(13)]

fig = plt.figure("Objective", dpi=200, figsize=(7, 2.5))
plt.plot(n_runs, devito, color='#be22d6', linewidth=1.0,
         label='devito on cpu')
plt.scatter(n_runs, devito, color='#be22d6', s=1.5)

plt.plot(n_runs, torch_cpu, color='#22c1d6', linewidth=1.0,
         label='torch on cpu')
plt.scatter(n_runs, torch_cpu, color='#22c1d6', s=1.5)

plt.plot(n_runs, torch_gpu, color='#aab304', linewidth=1.0,
         label='torch on gpu')
plt.scatter(n_runs, torch_gpu, color='#aab304', s=1.5)
plt.title("image size: " + r"$2048 \times 2048 \times 8$")
plt.legend(fontsize=8)
plt.xlabel("number of calls to conv")
plt.ylabel("wall-clock time (s)")
plt.grid()
plt.savefig('run-time-2048x2048x8.png' ,format='png', bbox_inches='tight',
            dpi=200, pad_inches=.05)
plt.close(fig)


# 2048 \times 2048 \times 4
devito = [10.130695104598999,
14.426070928573608,
22.96937918663025,
39.94618272781372,
74.02948784828186,
142.21478915214539]
# 277.6917886734009,
# 550.6099510192871,
# 1103.4632771015167,
# 2202.721147298813]

torch_cpu = [6.389679431915283,
9.40096378326416,
15.83627963066101,
29.051862955093384,
55.540682554244995,
108.46729922294617]


n_runs = [2**j for j in range(6)]

fig = plt.figure("Objective", dpi=200, figsize=(7, 2.5))
plt.plot(n_runs, devito, color='#be22d6', linewidth=1.0,
         label='devito on cpu')
plt.scatter(n_runs, devito, color='#be22d6', s=1.5)

plt.plot(n_runs, torch_cpu, color='#22c1d6', linewidth=1.0,
         label='torch on cpu')
plt.scatter(n_runs, torch_cpu, color='#22c1d6', s=1.5)

plt.title("image size: " + r"$16384 \times 16384 \times 2$")
plt.legend(fontsize=8)
plt.xlabel("number of calls to conv")
plt.ylabel("wall-clock time (s)")
plt.grid()
plt.savefig('run-time-16384x16384x2.png' ,format='png', bbox_inches='tight',
            dpi=200, pad_inches=.05)
plt.close(fig)

