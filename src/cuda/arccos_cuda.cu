/*
Compile as follows:
nvcc -arch=sm_90 -o arccos_cuda arccos_cuda.cu
*/

#include <cuda_runtime.h>
#include <iostream>

#define N 128 * 128
#define NUM_STREAMS 3

__global__ void compute_kernel(float* d_data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) d_data[idx] += value;
}

int main() {
    int size = N;
    size_t bytes = size * sizeof(float);

    float* h_data[NUM_STREAMS], *h_result[NUM_STREAMS];
    float* d_data[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];

    // Allocate host and device memory, create streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
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

        for (int j = 0; j < size; ++j) h_data[i][j] = static_cast<float>(j);
    }

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch operations in streams
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMemcpyAsync(d_data[i], h_data[i], bytes, cudaMemcpyHostToDevice, streams[i]); // HDx
        compute_kernel<<<blocks, threads, 0, streams[i]>>>(d_data[i], size, 1.0f);         // Kx
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel launch failed in stream " << i << ": " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        cudaMemcpyAsync(h_result[i], d_data[i], bytes, cudaMemcpyDeviceToHost, streams[i]); // DHx
    }

    // Wait for all streams to finish
    cudaDeviceSynchronize();

    // Verify result
    for (int i = 0; i < NUM_STREAMS; ++i) {
        bool correct = true;
        for (int j = 0; j < size; ++j) {
            if (h_result[i][j] != h_data[i][j] + 1.0f) {
                correct = false;
                std::cout << "Mismatch at index " << j << " in stream " << i << ": "
                          << h_result[i][j] << " != " << h_data[i][j] + 1.0f << std::endl;
                break;
            }
        }
        std::cout << "Stream " << i << ": " << (correct ? "Success" : "Failed") << std::endl;
    }

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaFreeHost(h_data[i]);
        cudaFreeHost(h_result[i]);
        cudaFree(d_data[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
