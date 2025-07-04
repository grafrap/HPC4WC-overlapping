import os
import numpy as np

# has to be called from the build folder (python reference/generate_arctan_data.py)

n_max = 1e8

if __name__=="__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(build_dir, "data")

    x = 2 * np.random.rand(int(n_max)) - 1
    ref_arccos = np.arccos(x)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(os.path.join(data_dir, "ref_data"), x=x, ref_single=ref_arccos)