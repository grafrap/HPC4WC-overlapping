#pragma once
#include <cuda_runtime.h>

__global__ void updateHaloKernel(double* field, int xsize, int ysize, int zsize, int halo);

__global__ void laplacianKernel(double* inField, double* outField, 
                               int xsize, int ysize, int zsize, int halo, int k_level);

__global__ void diffusionStepKernel(double* inField, double* tmp1Field, double* outField,
                                   int xsize, int ysize, int zsize, int halo, 
                                   double alpha, int k_level, bool isLastIter);
