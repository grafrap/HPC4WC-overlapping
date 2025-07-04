#include <cuda_runtime.h>
#include <chrono>

#include "cnpy.h"

// Debug flag (varification function)
#define DEBUG 0


// Define the floating-point type
using fType = float;

// CUDA kernel that computes the arccos of the element at the threads index
__global__ void compute_kernel(fType* d_data, int size);

// Function to initialize host data from refrence data
void init_h(fType* h_data, fType* h_result, fType* x, const fType* res, int i, int chunksize, int bytes);

// Function to initialize host data directly instead of reading from disk
void init_h_local(fType* h_data, fType* h_result, int i, int chunksize);

// Run the arccos computation using multiple CUDA streams
int run_arccos(int size, int num_streams, std::chrono::duration<double> &duration);

// Run stream operations
cudaError_t run_stream_operations(fType* h_data[], fType* d_data[], cudaStream_t streams[], int size_per_stream, int num_streams,
                                     int threads, int blocks);

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
bool verify_result(fType* h_result[], fType* h_data[], int size_per_stream, int num_streams);

// Debug version with additional argument
bool verify_result_debug(fType* h_result[], fType* h_data[], int size_per_stream, int num_streams, fType* h_data_debug[]);
