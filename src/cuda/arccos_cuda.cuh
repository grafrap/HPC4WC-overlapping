#include <cuda_runtime.h>
#include <iostream>
#include"cnpy.h"
#include<cstdlib>
#include<map>
#include<string>
#include<cstring>


// Define the floating-point type
using fType = float;

// CUDA kernel that computes the arccos of the element at the threads index
__global__ void compute_kernel(fType* d_data, int size, fType value);

// Function to initialize host data from refrence data
void init_h(fType* h_data, fType* h_result, const fType* x, const fType* res, int i, int chunksize, int bytes){

// Function to initialize result from reference data (arccos values)
void init_ref_result(float* ref_result, int size);

// Run the arccos computation using multiple CUDA streams
int run_arccos(int size, int num_streams);

// Run stream operations
cudaError_t run_stream_operations(float* h_data[], float* h_result[], float* d_data[], cudaStream_t streams[], int size_per_stream, int num_streams,
                                     int threads, int blocks);

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
void verify_result(float* h_result[], int size_per_stream, int num_streams);
