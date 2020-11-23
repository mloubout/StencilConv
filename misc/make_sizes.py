import numpy as np

def factor(n):
    r = np.arange(2, int(n ** 0.5) + 1).astype(np.int)
    x = r[np.mod(n, r) < n/100]
    return set(zip(x, n // x))

for N in range(5, 14):
    s = factor(2**(2*N))
    print("For n = 2^%d sizes are %s" %(N, s))