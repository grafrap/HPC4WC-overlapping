from arccos_gt4py import time_arccos_multiple_lengths
import os

npz_file = "build/data/ref_data.npz"
assert(os.path.isfile(npz_file))

sizes = [10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]

print("start timing")
time_arccos_multiple_lengths(npz_file, sizes, "measurements/measurement_gt4py_.csv", number=15, repeats=3)