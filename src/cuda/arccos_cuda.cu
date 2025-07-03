/*
Compile as follows:
nvcc -arch=sm_90 -o arccos_cuda arccos_cuda.cu
*/

#include <cuda_runtime.h>
#include <iostream>
#include"cnpy.h"
#include<cstdlib>
#include<map>
#include<string>
#include<cstring>
#include "arccos_cuda.cuh
#include<cmath>




__global__ void compute_kernel(fType* d_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_data[idx] = std::acos(d_data[idx]);
}

int run_arccos(int size, int num_streams) {
    // load data
    cnpy::NpyArray x_arr = cnpy::npz_load("data/ref_data.npz","x");
    // test if size < refernce data size
    int fullsize = x_arr.num_vals;
    assert(size <= fullsize);
    
    cnpy::NpyArray ref_arr = cnpy::npz_load("data/ref_data.npz","ref_single");
    // create pointers to data and convert to double if necessary????
    fType* x = x_arr.data<fType>();
    fType* ref = ref_arr.data<fType>();

    int size_per_stream = size / num_streams;
    if (size % num_streams != 0) {
        std::cerr << "Size must be divisible by number of streams." << std::endl;
        return;
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

        // for (int j = 0; j < size; ++j) h_data[i][j] = static_cast<fType>(j);
        init_h(h_data[i], h_result[i], x, ref, i, size_per_stream, bytes); // Initialize host data and reference data
    }

    int threads = 256;
    int blocks = (size_per_stream + threads - 1) / threads;

    // Launch operations in streams
    cudaError_t err = run_stream_operations(h_data, h_result, d_data, streams, size_per_stream, num_streams, threads, blocks);
    if (err != cudaSuccess) {
        std::cerr << "Cuda error after running stream operations: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    run_stream_operations(h_data, h_result, d_data, streams, size_per_stream, num_streams);
    // for (int i = 0; i < num_streams; ++i) {
    //     cudaMemcpyAsync(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice, streams[i]); // HDx
    //     compute_kernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], size, 1.0f);         // Kx
    //     cudaError_t err = cudaGetLastError();
    //     if (err != cudaSuccess) {
    //         std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
    //         return 1;
    //     }
    //     cudaMemcpyAsync(h_result[i], d_data[i], bytes, cudaMemcpyDeviceToHost, streams[i]); // DHx
    // }

    // Wait for all streams to finish
    // cudaDeviceSynchronize();

    // Verify result
    verify_result(h_result, size_per_stream, num_streams);
    // for (int i = 0; i < num_streams; ++i) {
    //     bool correct = true;
    //     for (int j = 0; j < size_per_stream; ++j) {
    //         if (h_result[i][j] != h_data[i][j] + 1.0f) {
    //             correct = false;
    //             std::cout << "Mismatch at index " << j << " in stream " << i << ": "
    //                       << h_result[i][j] << " != " << h_data[i][j] + 1.0f << std::endl;
    //             break;
    //         }
    //     }
    //     std::cout << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    // }

    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFreeHost(h_result[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }

}

// Function to initialize host data from refrence data
void init_h(fType* h_data, fType* h_result, const fType* x, const fType* res, int i, int chunksize, int bytes){
    // create pointer to start of subarray in x
    fType* srt_ptr = x + i * chunksize;
    // copy data
    std::memcpy(h_data, str_ptr, bytes);
    std::memcpy(h_result, str_ptr, bytes);
}

// Function to initialize result from reference data (arccos values)
//void init_ref_result(float* ref_result, int size) {

//}

// Run stream operations
cudaError_t run_stream_operations(float* h_data[], float* h_result[], float* d_data[], cudaStream_t streams[], int bytes_per_stream, int num_streams,
                                     int threads, int blocks) {
    // Loop through each stream and perform operations
    for (int i = 0; i < num_streams; ++i) {
        cudaError_t err = cudaMemcpyAsync(d_data[i], h_data[i], bytes_per_stream, cudaMemcpyHostToDevice, streams[i]);
        if (err != cudaSuccess) {
            std::cerr << "Memcpy (H2D) failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        compute_kernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], bytes_per_stream, 1.0f);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return err;
        }
        err = cudaMemcpyAsync(h_result[i], d_data[i], bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]);
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
void verify_result(float* h_result[], int size_per_stream, int num_streams) {

}

