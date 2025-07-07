#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include "stencil2d_helper/utils.h"



int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <size> <num_streams> <num_repetitions>" << std::endl;
        return 1;
    }

    long size = std::atoi(argv[1]);
    long num_streams = std::atoi(argv[2]);
    long num_repetitions = std::atoi(argv[3]);

    if (size <= 0 || num_streams <= 0 || num_repetitions <= 0) {
        std::cerr << "Size, number of streams and number of repetitions must be positive integers." << std::endl;
        return 1;
    }

    // Initialize the field
    Storage3D<double> field(size, size, size, 3);
    field.initialize();

    // initialize the Halo
    updateHalo(field);

    // allocate memory on the device with cudaMalloc
    Storage3D<double> d_field(size, size, size, 3);
    d_field.allocateDevice();

    // copy the field to the device
    cudaMemcpy(d_field.data(), field.data(), field.size() * sizeof(double), cudaMemcpyHostToDevice);

    // create streams
    cudaStream_t* streams = new cudaStream_t[num_streams];
    for (long i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // create timing variables
    cudaEvent_t start, stop;

    // run the laplapacian kernel

    return 0;
}