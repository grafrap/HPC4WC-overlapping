import gt4py.next as gtx
from gt4py.next.ffront.fbuiltins import arccos
import numpy as np
import timeit

I = gtx.Dimension("I")
IField = gtx.Field[gtx.Dims[I], gtx.float64]

@gtx.field_operator
def arccos_once(x: IField) -> IField:
    return arccos(x)

@gtx.field_operator
def arccos_twice(x: IField) -> IField:
    return arccos(arccos(x))

@gtx.field_operator
def arccos_three_times(x: IField) -> IField:
    return arccos(arccos(arccos(x)))

@gtx.field_operator
def arccos_four_times(x: IField) -> IField:
    return arccos(arccos(arccos(arccos(x))))

# backend = gtx.gtfn_cpu
backend = gtx.gtfn_gpu
gtx_arccos = arccos_once.with_backend(backend)

def load_data(npz_filepath):
    file = np.load(npz_filepath)
    x_np = file["x"]
    assert(len(x_np.shape) == 1)
    return x_np, file["ref"], x_np.shape[0]

def time_arccos_multiple_lengths(npz_file, sizes, filename_csv, number=15, repeats=3):
    """
    Time arccos inclusive data transfer to and from the gpu for various sizes of input data

    Parameters:
        npz_file (str): Path to the `.npz` file containing reference data.
        sizes (list or array-like): A list of input sizes to benchmark. Have to be within max_size of npz_file.
        filename_csv (str): Path to the CSV file where timing results will be saved.
        number (int, optional): Number of times to run the timed code per repeat (inner loop).
        repeats (int, optional): Number of repeats for timing (outer loop).
    """
    x_np, ref_np, max_size = load_data(npz_file)
    assert(np.max(sizes) <= max_size)
    measurements = np.empty((len(sizes),3)) # ["size", "time_mean", "time_std"]
    for i,size in enumerate(sizes):
        size = int(size)
        domain = gtx.domain({I: (0, size),})
        out_field = gtx.empty(domain=domain, dtype=x_np.dtype, allocator=backend)

        def benchmark():
            x = gtx.as_field(data=x_np[:size], domain=domain, allocator=backend)
            gtx_arccos(x=x, out=out_field, domain=domain)
            _ = out_field.asnumpy()

        print(f"start run with size {size}")
        times = timeit.repeat(benchmark, globals=globals(), repeat=repeats, number=number)
        measurements[i] = [size, np.mean(times), np.std(times)]
    np.savetxt(filename_csv, measurements, delimiter=",", header="size,time_mean,time_std")
    
if __name__=="__main__":
    x_np, ref_np, max_size = load_data("data/ref_data.npz")
    
    domain_all = gtx.domain({I: (0, max_size),})
    out_field = gtx.empty(domain=domain_all, dtype=x_np.dtype, allocator=backend)

    x = gtx.as_field(data=x_np, domain=domain_all, allocator=backend)
    gtx_arccos(x=x, out=out_field, domain=domain_all)
    
    assert(np.isclose(ref_np, out_field.asnumpy()).all())
    
    number = 15 #100 # inner loop reps
    repeats = 3 #10 # timings (outer loop)
    # warmups = 1
    times = timeit.repeat("x = gtx.as_field(data=x_np, domain=domain_all, allocator=backend); gtx_arccos(x=x, out=out_field, domain=domain_all); out_np = out_field.asnumpy()", globals=globals(), repeat=repeats, number=number)
    mean_time, std_time = np.mean(times), np.std(times)
    print(f"Average time per run: {mean_time / number:.6f} ± {std_time:.6f} seconds")
    