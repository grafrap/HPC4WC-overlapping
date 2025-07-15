/*
Compile as follows:
nvcc -arch=sm_90 -o arccos_cuda arccos_cuda.cu
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "arccos_cuda.cuh"


__global__ void compute_kernel_multiple(fType* d_data, int size, int num_arcos_calls) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int i = 0; i < num_arcos_calls; ++i) {
            d_data[idx] = std::acos(d_data[idx]);
        }
    }
}
 

int run_arccos(int num_arccos_calls, int size_per_stream, int num_streams, std::chrono::duration<double> &duration, fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], cudaStream_t streams[]) {

    // DEBUG: Copy h_data 
    // fType* h_data_debug[num_streams] = {nullptr};
    // for (int i = 0; i < num_streams; ++i) {
    //     h_data_debug[i] = (fType*)malloc(bytes);
    //     if (h_data_debug[i] == nullptr) {
    //         std::cerr << "Memory allocation failed for h_data_debug[" << i << "]" << std::endl;
    //         return 1;
    //     }
    //     std::memcpy(h_data_debug[i], h_data[i], bytes);
    // }

    // total number of threads = blocks * threaads = size per stream 
    int threads = THREADS_PER_BLOCK;
    int blocks = (size_per_stream + threads - 1) / threads;

    // Launch operations in streams
    auto start = std::chrono::high_resolution_clock::now();
    cudaError_t err = run_stream_operations(h_data, h_result, d_data, streams, num_arccos_calls, size_per_stream, num_streams, threads, blocks);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    duration = std::chrono::duration<double>(end - start);
    
    if (err != cudaSuccess) {
        std::cerr << "Cuda error after running stream operations: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Verify result
    bool correct_result;
    // if (DEBUG) {
    //     correct_result = verify_result_debug(h_result, h_data, size_per_stream, num_streams, h_data_debug);
    // } else {
    //     correct_result = verify_result(h_result, h_data, size_per_stream, num_streams);
    // }

    correct_result = verify_result(h_reference, h_result, size_per_stream, num_streams, num_arccos_calls);

    return correct_result ? 0 : 1; // Return 0 if all results are correct, otherwise return 1

}

int init_data(fType* h_data[], fType* h_result[], fType* h_reference[], fType* d_data[], size_t bytes, cudaStream_t streams[], int num_arccos_calls, int num_streams, int size_per_stream) {
    
    // Set up RNG
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<fType> dis(-1.0, 1.0);
    
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
        err = cudaHostAlloc(&h_reference[i], bytes, cudaHostAllocDefault);
        if (err != cudaSuccess) {
            std::cerr << "cudaHostAlloc failed for h_reference[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        err = cudaMalloc(&d_data[i], bytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed for d_data[" << i << "]: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        cudaStreamCreate(&streams[i]);

        // Initialize host data and reference data
        init_h_local(h_data[i], h_reference[i], num_arccos_calls, size_per_stream, gen, dis);
    }
    return 0; // Return 0 on success
}

void init_h_local(fType* h_data, fType* h_result, int num_arccos_calls, int chunksize, std::mt19937 &gen, std::uniform_real_distribution<fType> &dis) {
    
    // Initialize data with random values in the range [-1, 1]
    for (int j = 0; j < chunksize; ++j) {
        h_data[j] = dis(gen); // Example initialization
        
        // Precompute the expected result
        h_result[j] = std::acos(h_data[j]);
        for (int k = 1; k < num_arccos_calls; ++k) {
            h_result[j] = std::acos(h_result[j]); // Apply arccos multiple times
        }
    }
}

// Run stream operations
cudaError_t run_stream_operations(fType* h_data[], fType* h_result[], fType* d_data[], cudaStream_t streams[], int num_arccos_calls, int size_per_stream, int num_streams,
                                     int threads, int blocks) {
    // Loop through each stream and perform operations
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaMemcpyAsync(d_data[i], h_data[i], size_per_stream * sizeof(fType), cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (H2D) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        compute_kernel_multiple<<<blocks, threads, 0, streams[i]>>>(d_data[i], size_per_stream, num_arccos_calls);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        err = cudaMemcpyAsync(h_result[i], d_data[i], size_per_stream * sizeof(fType), cudaMemcpyDeviceToHost, streams[i]);
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
bool verify_result(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams, int num_arccos_calls) {
    // fType adaptive_tol = TOL * (1 + num_arccos_calls); 
    for (int i = 0; i < num_streams; ++i) {
        for (int j = 0; j < size_per_stream; ++j) {
            if (std::fabs(h_reference[i][j] - h_result[i][j]) > 1e-3) {
                std::cerr << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_reference[i][j] << " != " << h_result[i][j] << " with a difference of " << std::fabs(h_reference[i][j] - h_result[i][j]) << std::endl;
                return false; // Early exit on first mismatch
            }
        }
        // std::cerr << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    }
    return true; // All streams verified successfully
}

// Verify the result of the arccos computation (return bool?) (call init_ref_result for the reference result)
bool verify_result_debug(fType* h_reference[], fType* h_result[], int size_per_stream, int num_streams, fType* h_data_debug[]) {
    for (int i = 0; i < num_streams; ++i) {
        for (int j = 0; j < size_per_stream; ++j) {
            if (std::fabs(h_reference[i][j] - h_result[i][j]) > TOL) {
                std::cerr << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_reference[i][j] << " != " << h_result[i][j] << ", x = " << h_data_debug[i][j] << std::endl;
                return false; // Early exit on first mismatch
            }
        }
        // std::cerr << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    }
    return true; // All streams verified successfully
}

void cleanup(fType* h_data[], fType* h_result[], fType* h_refernce[], fType* d_data[], cudaStream_t streams[], int num_streams) {
    
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFreeHost(h_result[i]);
        cudaFreeHost(h_refernce[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
        // // DEBUG: Free debug data
        // free(h_data_debug[i]);
    }
}

