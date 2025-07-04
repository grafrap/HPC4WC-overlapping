/*
Compile as follows:
nvcc -arch=sm_90 -o arccos_cuda arccos_cuda.cu
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <map>
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>
#include <random>

#include "cnpy.h"
#include "arccos_cuda.cuh"

#define TOL 1e-5 // Define a tolerance for floating-point comparison




__global__ void compute_kernel(fType* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_data[idx] = std::acos(d_data[idx]);
}

int run_arccos(int size, int num_streams, std::chrono::duration<double> &duration) {
    // // load data
    // cnpy::NpyArray x_arr = cnpy::npz_load("data/ref_data.npz","x");
    // // test if size < refernce data size
    // int fullsize = x_arr.num_vals;
    // assert(size <= fullsize);
    
    // cnpy::NpyArray ref_arr = cnpy::npz_load("data/ref_data.npz","ref_single");
    // // create pointers to data and convert to double if necessary????
    // fType* x = x_arr.data<fType>();
    // fType* ref = ref_arr.data<fType>();

    int size_per_stream = size / num_streams;
    if (size % num_streams != 0) {
        std::cerr << "Size must be divisible by number of streams." << std::endl;
        return 1;
    }
    size_t bytes = size_per_stream * sizeof(fType);

    fType* h_data[num_streams], *h_result[num_streams];
    fType* d_data[num_streams];
    cudaStream_t streams[num_streams];

    // Allocate host and device memory, create streams
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaHostAlloc(&h_data[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        err = cudaHostAlloc(&h_result[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        cudaMalloc(&d_data[i], bytes);
        cudaStreamCreate(&streams[i]);

        // Initialize host data and reference data
        // init_h(h_data[i], h_result[i], x, ref, i, size_per_stream, bytes); 
        init_h_local(h_data[i], h_result[i], i, size_per_stream);
    }

    // DEBUG: Copy h_data 
    fType* h_data_debug[num_streams] = {nullptr};
    for (int i = 0; i < num_streams; ++i) {
        h_data_debug[i] = (fType*)malloc(bytes);
        if (h_data_debug[i] == nullptr) {
            std::cerr << "Memory allocation failed for h_data_debug[" << i << "]" << std::endl;
            return 1;
        }
        std::memcpy(h_data_debug[i], h_data[i], bytes);
    }

    int threads = 256;
    int blocks = (size_per_stream + threads - 1) / threads;

    // Launch operations in streams
    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t err = run_stream_operations(h_data, d_data, streams, size_per_stream, num_streams, threads, blocks);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    duration = std::chrono::duration<double>(end - start);
    
    if (err != cudaSuccess) {
        std::cerr << "Cuda error after running stream operations: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Verify result
    bool correct_result;
    if (DEBUG) {
        correct_result = verify_result_debug(h_result, h_data, size_per_stream, num_streams, h_data_debug);
    } else {
        correct_result = verify_result(h_result, h_data, size_per_stream, num_streams);
    }

    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFreeHost(h_result[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
        // DEBUG: Free debug data
        free(h_data_debug[i]);
    }

    return correct_result ? 0 : 1; // Return 0 if all results are correct, otherwise return 1

}

// Function to initialize host data from refrence data
void init_h(fType* h_data, fType* h_result, fType* x, const fType* res, int i, int chunksize, int bytes) {
    // create pointer to start of subarray in x
    fType* str_ptr = x + i * chunksize;
    // copy data
    std::memcpy(h_data, str_ptr, bytes);
    std::memcpy(h_result, str_ptr, bytes);
}

void init_h_local(fType* h_data, fType* h_result, int i, int chunksize) {
    
    // Set up RNG
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<fType> dis(-1.0, 1.0);
    
    // Initialize data with random values in the range [-1, 1]
    for (int j = 0; j < chunksize; ++j) {
        h_data[j] = dis(gen); // Example initialization
        h_result[j] = std::acos(h_data[j]); // Precompute the expected result
    }
}

// Run stream operations
cudaError_t run_stream_operations(fType* h_data[], fType* d_data[], cudaStream_t streams[], int size_per_stream, int num_streams,
                                     int threads, int blocks) {
    // Loop through each stream and perform operations
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaMemcpyAsync(d_data[i], h_data[i], size_per_stream * sizeof(fType), cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (H2D) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        compute_kernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], size_per_stream);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        err = cudaMemcpyAsync(h_data[i], d_data[i], size_per_stream * sizeof(fType), cudaMemcpyDeviceToHost, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (D2H) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();

    // Check for errors after synchronization
    return cudaGetLastError();
}

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
bool verify_result(fType* h_result[], fType* h_data[], int size_per_stream, int num_streams) {
    for (int i = 0; i < num_streams; ++i) {
        bool correct = true;
        for (int j = 0; j < size_per_stream; ++j) {
            if (std::fabs(h_result[i][j] - h_data[i][j]) > TOL) {
                correct = false;
                std::cout << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_result[i][j] << " != " << h_data[i][j] << std::endl;
                return false; // Early exit on first mismatch
            }
        }
        std::cout << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    }
    return true; // All streams verified successfully
}

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
bool verify_result_debug(fType* h_result[], fType* h_data[], int size_per_stream, int num_streams, fType* h_data_debug[]) {
    for (int i = 0; i < num_streams; ++i) {
        bool correct = true;
        for (int j = 0; j < size_per_stream; ++j) {
            if (std::fabs(h_result[i][j] - h_data[i][j]) > TOL) {
                correct = false;
                std::cout << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_result[i][j] << " != " << h_data[i][j] << ", x = " << h_data_debug[i][j] << std::endl;
                return false; // Early exit on first mismatch
            }
        }
        std::cout << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    }
    return true; // All streams verified successfully
}

