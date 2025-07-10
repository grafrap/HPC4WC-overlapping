#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "stencil2d_helper/utils.h"
#include "stencil2d_helper/stencil_kernels.cuh"

void apply_diffusion_gpu_streams(Storage3D<double> &inField, Storage3D<double> &outField,
                               double alpha, unsigned numIter, int x, int y, int z,
                               int halo, int numStreams = 2) {
    
    // Allocate device memory
    inField.allocateDevice();
    outField.allocateDevice();
    
    // Create temporary field for GPU computation
    Storage3D<double> tmp1Field(x, y, z, halo); 
    tmp1Field.allocateDevice();
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Copy input data to device (initial setup)
    inField.copyToDevice();
    
    // Calculate work distribution per stream
    int zPerStream = (z + numStreams - 1) / numStreams;
    
    // Set up CUDA execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((x + halo * 2 + blockSize.x - 1) / blockSize.x,
                  (y + halo * 2 + blockSize.y - 1) / blockSize.y);
    
    // Halo update configuration (can be done on stream 0)
    dim3 haloBlockSize(16, 16, 1);
    dim3 haloGridSize((inField.xSize() + haloBlockSize.x - 1) / haloBlockSize.x,
                     (inField.ySize() + haloBlockSize.y - 1) / haloBlockSize.y,
                     (z + haloBlockSize.z - 1) / haloBlockSize.z);
    
    for (unsigned iter = 0; iter < numIter; ++iter) {
        // GPU halo update on first stream
        updateHaloKernel2D<<<haloGridSize, haloBlockSize, 0, streams[0]>>>(
            inField.deviceData(), inField.xSize(), inField.ySize(), inField.zMax(), halo
        );
        
        // Wait for halo update to complete before starting computation
        cudaStreamSynchronize(streams[0]);
        
        // Launch diffusion kernels on multiple streams
        for (int streamId = 0; streamId < numStreams; ++streamId) {
            int startK = streamId * zPerStream;
            int endK = std::min(startK + zPerStream, z);
            
            for (int k = startK; k < endK; ++k) {
                diffusionStepKernel<<<gridSize, blockSize, 0, streams[streamId]>>>(
                    inField.deviceData(), outField.deviceData(), tmp1Field.deviceData(),
                    inField.xSize(), inField.ySize(), inField.zMax(), k, halo, alpha
                );
            }
        }
        
        // Synchronize all streams before next iteration
        for (int i = 0; i < numStreams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // If not the last iteration, copy output back to input
        if (iter < numIter - 1) {
            cudaMemcpyAsync(inField.deviceData(), outField.deviceData(), 
                           inField.size() * sizeof(double), cudaMemcpyDeviceToDevice, 
                           streams[0]);
            cudaStreamSynchronize(streams[0]);
        }
    }
    
    // Cleanup streams
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    
    // Copy final result back to host
    outField.copyFromDevice();
}

void reportTime(const Storage3D<double> &storage, int nIter, double diff, int nStreams = 1) {
    std::cout << "# ranks nx ny nz num_iter time\n";
    int size = 1; // Assuming single GPU
    std::cout << size << ", " << storage.xMax() - storage.xMin() << ", "
              << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
              << nIter << ", " << diff << ", " << nStreams << "\n" ;
}

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] << " -nx <x> -ny <y> -nz <z> -iter <iterations> -streams <numStreams>" << std::endl;
        return 1;
    }
    
    int x = atoi(argv[2]);
    int y = atoi(argv[4]);
    int z = atoi(argv[6]);
    int iter = atoi(argv[8]);
    int numStreams = atoi(argv[10]); // Number of streams from command line
    int nHalo = 3;
    
    assert(x > 0 && y > 0 && z > 0 && iter > 0);
    
    Storage3D<double> input(x, y, z, nHalo);
    input.initialize();
    Storage3D<double> output(x, y, z, nHalo);
    output.initialize();

    double alpha = 1. / 32.;

    // Write initial field
    std::ofstream fout;
    fout.open("in_field_streams.dat", std::ios::binary | std::ofstream::trunc);
    input.writeFile(fout);
    fout.close();

#ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
#endif
    auto start = std::chrono::steady_clock::now();

    apply_diffusion_gpu_streams(input, output, alpha, iter, x, y, z, nHalo, numStreams);

    auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif

    updateHalo(output);
    fout.open("out_field_streams.dat", std::ios::binary | std::ofstream::trunc);
    output.writeFile(fout);
    fout.close();

    auto diff = end - start;
    double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
    reportTime(output, iter, timeDiff, numStreams);

    return 0;
}


// second version for the gpu streams
void apply_diffusion_gpu_streams_advanced(Storage3D<double> &inField, Storage3D<double> &outField,
                                         double alpha, unsigned numIter, int x, int y, int z,
                                         int halo, int numStreams = 3) {
    
    // Allocate device memory
    inField.allocateDevice();
    outField.allocateDevice();
    
    Storage3D<double> tmp1Field(x, y, z, halo); 
    tmp1Field.allocateDevice();
    
    // Create CUDA streams
    std::vector<cudaStream_t> streams(numStreams);
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
    
    // Allocate pinned host memory for better transfer performance
    double* h_pinned_in;
    double* h_pinned_out;
    size_t totalSize = inField.size() * sizeof(double);
    
    cudaMallocHost(&h_pinned_in, totalSize);
    cudaMallocHost(&h_pinned_out, totalSize);
    
    // Copy initial data to pinned memory
    memcpy(h_pinned_in, inField.data(), totalSize);
    
    // Initial async copy to device
    cudaMemcpyAsync(inField.deviceData(), h_pinned_in, totalSize, 
                    cudaMemcpyHostToDevice, streams[0]);
    
    // Calculate work distribution
    int zPerStream = (z + numStreams - 2) / (numStreams - 1); // Reserve one stream for transfers
    
    dim3 blockSize(16, 16);
    dim3 gridSize((x + halo * 2 + blockSize.x - 1) / blockSize.x,
                  (y + halo * 2 + blockSize.y - 1) / blockSize.y);
    
    dim3 haloBlockSize(16, 16, 1);
    dim3 haloGridSize((inField.xSize() + haloBlockSize.x - 1) / haloBlockSize.x,
                     (inField.ySize() + haloBlockSize.y - 1) / haloBlockSize.y,
                     (z + haloBlockSize.z - 1) / haloBlockSize.z);
    
    for (unsigned iter = 0; iter < numIter; ++iter) {
        // Wait for previous transfer to complete
        cudaStreamSynchronize(streams[0]);
        
        // GPU halo update
        updateHaloKernel2D<<<haloGridSize, haloBlockSize, 0, streams[1]>>>(
            inField.deviceData(), inField.xSize(), inField.ySize(), inField.zMax(), halo
        );
        
        cudaStreamSynchronize(streams[1]);
        
        // Launch computation on multiple streams (excluding transfer stream)
        for (int streamId = 1; streamId < numStreams; ++streamId) {
            int startK = (streamId - 1) * zPerStream;
            int endK = std::min(startK + zPerStream, z);
            
            for (int k = startK; k < endK; ++k) {
                diffusionStepKernel<<<gridSize, blockSize, 0, streams[streamId]>>>(
                    inField.deviceData(), outField.deviceData(), tmp1Field.deviceData(),
                    inField.xSize(), inField.ySize(), inField.zMax(), k, halo, alpha
                );
            }
        }
        
        // Synchronize computation streams
        for (int i = 1; i < numStreams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        // Prepare for next iteration or final result
        if (iter < numIter - 1) {
            // Async copy output to input for next iteration
            cudaMemcpyAsync(inField.deviceData(), outField.deviceData(), 
                           totalSize, cudaMemcpyDeviceToDevice, streams[0]);
        } else {
            // Final async copy to host
            cudaMemcpyAsync(h_pinned_out, outField.deviceData(), totalSize, 
                           cudaMemcpyDeviceToHost, streams[0]);
        }
    }
    
    // Wait for final transfer and copy to output
    cudaStreamSynchronize(streams[0]);
    memcpy(outField.data(), h_pinned_out, totalSize);
    
    // Cleanup
    for (int i = 0; i < numStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
}