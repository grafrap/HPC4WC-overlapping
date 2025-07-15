#include <cuda_runtime.h>
#include <chrono>
#include <random>

#include "cnpy.h"

// Number of threads used per block on GPU
#define THREADS_PER_BLOCK 256

// RNG seed for random number generation
#define SEED 42

// Define a tolerance for floating-point comparison
#define TOL 1e-1

// Debug flag (varification function)
#define DEBUG 0


// Define the floating-point type
using fType = float;

// CUDA kernel that computes the arccos of the element at the threads index
__global__ void compute_kernel_once(fType* d_data, int size);

// CUDA kernel that computes the arccos(arccos) of the element at the threads index
__global__ void compute_kernel_multiple(fType* d_data, int size, int num_arcos_calls);

// Initialize all host and device arrays and the streams
int init_data(fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], size_t bytes, cudaStream_t streams[], int num_arccos_calls, int num_streams, int size_per_stream);

// Function to initialize host data directly instead of reading from disk
void init_h_local(fType* h_data, fType* h_result, int i, int num_arccos_calls, int chunksize, std::mt19937 &gen, std::uniform_real_distribution<fType> &dis);

// Run the arccos computation using multiple CUDA streams
int run_arccos(int num_arccos_calls, int size, int num_streams, std::chrono::duration<double> &duration, fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], cudaStream_t streams[]);

// Run stream operations
cudaError_t run_stream_operations(fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[], int num_arccos_calls, int size_per_stream, int num_streams,
                                     int threads, int blocks);

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
bool verify_result(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams);

// Debug version with additional argument
bool verify_result_debug(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams, fType* h_data_debug[]);

// Delete all allocated memory and destroy all streams
void cleanup(fType* h_data[], fType* h_result[], fType* h_refernce[], fType* d_data[], cudaStream_t streams[], int num_streams);