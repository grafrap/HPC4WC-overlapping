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

void apply_diffusion_gpu(Storage3D<double> &inField, Storage3D<double> &outField,
                        double alpha, unsigned numIter, int x, int y, int z,
                        int halo) {

    // Allocate device memory
    inField.allocateDevice();
    outField.allocateDevice();
    
    // Create temporary field for GPU computation - needs to be same size as input
    Storage3D<double> tmp1Field(x, y, z, halo); 
    tmp1Field.allocateDevice();
    
    // Copy input data to device
    inField.copyToDevice();
    
    // Check for CUDA errors after setup
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA setup error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // Set up CUDA execution configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((x + halo * 2 + blockSize.x - 1) / blockSize.x,
                  (y + halo * 2 + blockSize.y - 1) / blockSize.y);
    
    std::cout << "Grid size: " << gridSize.x << "x" << gridSize.y << std::endl;
    std::cout << "Block size: " << blockSize.x << "x" << blockSize.y << std::endl;
    dim3 haloBlockSize(16, 16, 1);
    dim3 haloGridSize((x + haloBlockSize.x - 1) / haloBlockSize.x,
                     (y + haloBlockSize.y - 1) / haloBlockSize.y,
                     (z + haloBlockSize.z - 1) / haloBlockSize.z);
    
    for (unsigned iter = 0; iter < numIter; ++iter) {
        // GPU halo update - much faster than CPU version!
        updateHaloKernel2D<<<haloGridSize, haloBlockSize>>>(
            inField.deviceData(), inField.xSize(), inField.ySize(), inField.zMax(), halo
        );
        
        cudaDeviceSynchronize(); // Ensure halo update completes
        
        for (int k = 0; k < z; ++k) {
            diffusionStepKernel<<<gridSize, blockSize>>>(
                inField.deviceData(), outField.deviceData(), tmp1Field.deviceData(),
                inField.xSize(), inField.ySize(), inField.zMax(), k, halo, alpha
            );
        }
        
        // If not the last iteration, copy output back to input
        if (iter < numIter - 1) {
            cudaMemcpy(inField.deviceData(), outField.deviceData(), 
                      inField.size() * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }
    
    // Copy final result back to host
    outField.copyFromDevice();
}

void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
    std::cout << "# ranks nx ny nz num_iter time\ndata = np.array( [ \\\n";
    int size = 1; // Assuming single GPU
    std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
              << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
              << nIter << ", " << diff << "],\n";
    std::cout << "] )" << std::endl;
}

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif
    if (argc != 9) {
        std::cerr << "Usage: " << argv[0] << " -nx <x> -ny <y> -nz <z> -iter <iterations>" << std::endl;
        return 1;
    }
    
    int x = atoi(argv[2]);
    int y = atoi(argv[4]);
    int z = atoi(argv[6]);
    int iter = atoi(argv[8]);
    int nHalo = 3;
    
    assert(x > 0 && y > 0 && z > 0 && iter > 0);
    
    Storage3D<double> input(x, y, z, nHalo);
    input.initialize();
    Storage3D<double> output(x, y, z, nHalo);
    output.initialize();

    double alpha = 1. / 32.;

    // Write initial field
    std::ofstream fout;
    fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
    input.writeFile(fout);
    fout.close();

#ifdef CRAYPAT
    PAT_record(PAT_STATE_ON);
#endif
    auto start = std::chrono::steady_clock::now();

    apply_diffusion_gpu(input, output, alpha, iter, x, y, z, nHalo);

    auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
    PAT_record(PAT_STATE_OFF);
#endif

    updateHalo(output);
    fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
    output.writeFile(fout);
    fout.close();

    auto diff = end - start;
    double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
    reportTime(output, iter, timeDiff);

    return 0;
}
