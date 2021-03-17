[![DOI](https://zenodo.org/badge/315379853.svg)](https://zenodo.org/badge/latestdoi/315379853)

# StencilConv

Stencil based convolution with [Devito](https://github.com/devitocodes/devito/commits/master)

# Content

- [benchmark_cpu_2d](https://github.com/mloubout/StencilConv/tree/master/benchmark_cpu_2d): Contains the 2D benchmark devito vs torch scripts and results (Ali what was the AWS hardware?)
- [compare_scrips](https://github.com/mloubout/StencilConv/tree/master/compare_scripts): Implementation fo the 2D and 3D multi-channel convolution and comparison against pytorch for correctness of the result.


# Basic testrun

Basic 2D convolution with multi chanels, this is an old script need to rerun the correct `test_nch_3d.py` instead

- Batchsize can be added
- Can think about stride later

Runs with and without MPI, near same result
> halo exachange missing at the begginging so slightly different result at edge of domain decomp

Correction, everythign is fine, we get slightly different results when runnign single precsion, but exact same results when running double precision
so I think we in business.

Thats a 3e-4 relative difference below in single precision, can change float32, float64 at top of the script

```bash
➜  StencilConv git:(master) ✗ python3 test.py                                              
Operator `Kernel` run in 0.03 s
64777.707
➜  StencilConv git:(master) ✗  DEVITO_MPI=1 mpirun -n 2 python3 test.py
Operator `Kernel` run in 0.03 s
Operator `Kernel` run in 0.03 s
64799.973
64799.973
```
