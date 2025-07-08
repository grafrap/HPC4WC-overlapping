#pragma once
#include <cuda_runtime.h>

__global__ void updateHaloKernel(double* field, int xsize, int ysize, int zsize, int halo);

__global__ void diffusionStepKernel(double* inField, double* outField, double* tmp1Field,
                                    int xsize, int ysize, int zsize, int k_level, int halo, double alpha);

__global__ void updateHaloKernel2D(double* field, int xsize, int ysize, int zsize, int halo);