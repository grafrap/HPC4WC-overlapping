import sys
from arccos_gt4py import test_arccos
import os
import numpy as np

print(f"{sys.argv[0]} started\n")

sys.setrecursionlimit(5000)

# ncalls and sizes as in run-arccos_cuda.sh
ncalls = 2**np.arange(10)
sizes = 2**np.arange(3,30,2)

smax = sizes.max()

all_close = True
for n in ncalls:
    try:
        test_arccos(n, smax)
    except Exception as e:
        all_close = False

if all_close:
    print("\nTest successful!\n")
else:
    print("\nTest failed!\n")