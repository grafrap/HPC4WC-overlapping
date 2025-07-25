import gt4py.next as gtx
from gt4py.next import broadcast
from gt4py.next.ffront.fbuiltins import arccos, minimum, maximum
import numpy as np
import timeit
import sys
import os

I = gtx.Dimension("I")
IField = gtx.Field[gtx.Dims[I], gtx.float64]

@gtx.field_operator # 2^0
def arccos_rescaled(x: IField) -> IField:
    res = broadcast(2.0 / 3.141592653589793, (I,)) * arccos(x) - broadcast(1.0, (I,))
    return res

@gtx.field_operator # 2^1
def arccos_twice(x: IField) -> IField:
    return arccos_rescaled(arccos_rescaled(x))

@gtx.field_operator # 2^2
def arccos_four_times(x: IField) -> IField:
    return arccos_rescaled(arccos_rescaled(arccos_rescaled(arccos_rescaled(x))))

@gtx.field_operator # 2^3
def arccos_2tt3_times(x: IField) -> IField:
    return arccos_four_times(arccos_four_times(x))

@gtx.field_operator # 2^4
def arccos_2tt4_times(x: IField) -> IField:
    return arccos_2tt3_times(arccos_2tt3_times(x))

@gtx.field_operator # 2^5
def arccos_2tt5_times(x: IField) -> IField:
    return arccos_2tt4_times(arccos_2tt4_times(x))

@gtx.field_operator # 2^6
def arccos_2tt6_times(x: IField) -> IField:
    return arccos_2tt5_times(arccos_2tt5_times(x))

# @gtx.field_operator # 2^7
# def arccos_2tt7_times(x: IField) -> IField:
#     return arccos_2tt6_times(arccos_2tt6_times(x))

# @gtx.field_operator # 2^8
# def arccos_2tt8_times(x: IField) -> IField:
#     return arccos_2tt7_times(arccos_2tt7_times(x))

# @gtx.field_operator # 2^9
# def arccos_2tt9_times(x: IField) -> IField:
#     return arccos_2tt8_times(arccos_2tt8_times(x))

# choose backend based on environment varible if provided
env_var_backend = os.environ.get("USE_BACKEND")
if env_var_backend == "GPU":
    backend = gtx.gtfn_gpu
elif env_var_backend == "CPU":
    backend = gtx.gtfn_cpu
elif env_var_backend is None:  # default case
    backend = gtx.gtfn_cpu
    # backend = gtx.gtfn_gpu
else:
    print(f"Invalid value '{env_var_backend}' in environment variable 'USE_BACKEND'")
    sys.exit(1)

gtx_arccos1 = arccos_rescaled.with_backend(backend)
gtx_arccos2 = arccos_twice.with_backend(backend)
gtx_arccos4 = arccos_four_times.with_backend(backend)
gtx_arccos8 = arccos_2tt3_times.with_backend(backend)
gtx_arccos2tt4 = arccos_2tt4_times.with_backend(backend)
gtx_arccos2tt5 = arccos_2tt5_times.with_backend(backend)
gtx_arccos2tt6 = arccos_2tt6_times.with_backend(backend)
# gtx_arccos2tt7 = arccos_2tt7_times.with_backend(backend)
# gtx_arccos2tt8 = arccos_2tt8_times.with_backend(backend)
# gtx_arccos2tt9 = arccos_2tt9_times.with_backend(backend)

def arccos2tt7(x, out, domain):
    temp_field = gtx.empty(domain=domain, dtype=x.dtype, allocator=backend)
    gtx_arccos2tt6(x=x, out=temp_field, domain=domain)
    gtx_arccos2tt6(x=temp_field, out=out, domain=domain)
    
def arccos2tt8(x, out, domain):
    temp_field = gtx.empty(domain=domain, dtype=x.dtype, allocator=backend)
    gtx_arccos2tt6(x=x, out=temp_field, domain=domain)
    for _ in range(4-2):
        gtx_arccos2tt6(x=temp_field, out=temp_field, domain=domain)
    gtx_arccos2tt6(x=temp_field, out=out, domain=domain)
    
def arccos2tt9(x, out, domain):
    temp_field = gtx.empty(domain=domain, dtype=x.dtype, allocator=backend)
    gtx_arccos2tt6(x=x, out=temp_field, domain=domain)
    for _ in range(8-2):
        gtx_arccos2tt6(x=temp_field, out=temp_field, domain=domain)
    gtx_arccos2tt6(x=temp_field, out=out, domain=domain)

fct_dict = {2**0: gtx_arccos1,
            2**1: gtx_arccos2,
            2**2: gtx_arccos4,
            2**3: gtx_arccos8,
            2**4: gtx_arccos2tt4,
            2**5: gtx_arccos2tt5,
            2**6: gtx_arccos2tt6,
            # 2**7: gtx_arccos2tt7,
            # 2**8: gtx_arccos2tt8,
            # 2**9: gtx_arccos2tt9
            2**7: arccos2tt7,
            2**8: arccos2tt8,
            2**9: arccos2tt9
           }

def get_test_fct(num_arccos_calls):
    if not num_arccos_calls in fct_dict.keys():
        print(f"<num_arccos_calls> = {num_arccos_calls}, but there is no function for this number of calls available", file=sys.stderr)
        print(f"dtype num_arccos_calls: {type(num_arccos_calls)} vs. dtype keys: {type(2**4)} resp. {list(fct_dict.keys())[3]}")
        print(f"valid values would be: ", list(fct_dict.keys()))
        # print("Usage: " << sys.argv[0] << "<num_arccos_calls> <size> <num_streams> <num_repetitions>", file=sys.stderr)
        sys.exit(1)
    return fct_dict[num_arccos_calls]

def gen_data(size):
    x = 2 * np.random.rand(size) - 1
    # ref_arccos = np.arccos(x)
    return x

def time_arccos(num_arccos_calls, size, number=1, repeats=10, do_print=True):
    """
    Time arccos inclusive data transfer to and from the gpu for fixed size

    Parameters:
        # TODO
    

        number (int, optional): Number of times to run the timed code per repeat (inner loop).
        repeats (int, optional): Number of repeats for timing (outer loop).
    """

    test_fct = get_test_fct(int(num_arccos_calls))
    
    x_np = gen_data(size)

    domain = gtx.domain({I: (0, size),})
    out_field = gtx.empty(domain=domain, dtype=x_np.dtype, allocator=backend)

    def benchmark():
        x = gtx.as_field(data=x_np[:size], domain=domain, allocator=backend)
        test_fct(x=x, out=out_field, domain=domain)
        _ = out_field.asnumpy()

    times = timeit.repeat(benchmark, globals=globals(), repeat=repeats, number=number)
    avg_time = np.mean(times)
    if do_print:
        # Calls, Size, NUM_STREAMS (unknown for gt4py => -1), Time
        print(f"### {num_arccos_calls} {size} {-1} {avg_time}")
    return (num_arccos_calls, size, avg_time)
    
if __name__=="__main__":
    if len(sys.argv) >= 3:
        num_arccos_calls = int(sys.argv[1])
        size = int(float(sys.argv[2]))
        time_arccos(num_arccos_calls, size, number=1, repeats=10)

    else:
        print("Usage for multiple sizes and arccos calls: ", sys.argv[0], "<num_arccos_calls> <size>", file=sys.stderr)
        # print("Usage: " << sys.argv[0] << "<num_arccos_calls> <size> <num_streams> <num_repetitions>", file=sys.stderr)

        print("\nThere were not enough arguments -> Run only for size 10^8 and a single arccos call:")
        size = int(1e8)
        test_fct = get_test_fct(1)
        x_np = gen_data(size)
        domain = gtx.domain({I: (0, size),})
        out_field = gtx.empty(domain=domain, dtype=x_np.dtype, allocator=backend)

        x = gtx.as_field(data=x_np, domain=domain, allocator=backend)
        test_fct(x=x, out=out_field, domain=domain)
        
        number = 1 #100 # inner loop reps
        repeats = 10 # timings (outer loop)
        # warmups = 1
        times = timeit.repeat("x = gtx.as_field(data=x_np, domain=domain, allocator=backend); test_fct(x=x, out=out_field, domain=domain); out_np = out_field.asnumpy()", globals=globals(), repeat=repeats, number=number)
        avg_time, std_time = np.mean(times), np.std(times)
        print(f"Average time per run: {avg_time / number:.6f} ± {std_time:.6f} seconds")