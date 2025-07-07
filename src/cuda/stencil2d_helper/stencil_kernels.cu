#include "stencil_kernels.cuh"

__device__ double device_access(double* data, int i, int j, int k, int xsize, int ysize) {
    return data[i + j * xsize + k * xsize * ysize];
}

__device__ void device_set(double* data, int i, int j, int k, int xsize, int ysize, double value) {
    data[i + j * xsize + k * xsize * ysize] = value;
}

__global__ void updateHaloKernel(double* field, int xsize, int ysize, int zsize, int halo) {
    int xInterior = xsize - 2 * halo;
    int yInterior = ysize - 2 * halo;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_halo_points = xInterior * halo * zsize * 2 + // top/bottom edges
                           yInterior * halo * zsize * 2;   // left/right edges
    
    if (idx >= total_halo_points) return;
    
    // This is a simplified version - you may need to implement more sophisticated indexing
    // for better performance
}

__global__ void laplacianKernel(double* inField, double* outField, 
                               int xsize, int ysize, int zsize, int halo, int k_level) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + halo;
    int j = blockIdx.y * blockDim.y + threadIdx.y + halo;
    
    if (i >= xsize - halo || j >= ysize - halo) return;
    
    int idx = i + j * xsize + k_level * xsize * ysize;
    
    outField[idx] = -4.0 * inField[idx] + 
                    inField[idx - 1] +           // i-1
                    inField[idx + 1] +           // i+1  
                    inField[idx - xsize] +       // j-1
                    inField[idx + xsize];        // j+1
}

__global__ void diffusionStepKernel(double* inField, double* tmp1Field, double* outField,
                                   int xsize, int ysize, int zsize, int halo, 
                                   double alpha, int k_level, bool isLastIter) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + halo;
    int j = blockIdx.y * blockDim.y + threadIdx.y + halo;
    
    if (i >= xsize - halo || j >= ysize - halo) return;
    
    int idx = i + j * xsize + k_level * xsize * ysize;
    int tmp_idx = i + j * xsize; // tmp1Field is 2D
    
    // First laplacian - store in tmp1Field
    tmp1Field[tmp_idx] = -4.0 * inField[idx] + 
                        inField[idx - 1] +           // i-1
                        inField[idx + 1] +           // i+1  
                        inField[idx - xsize] +       // j-1
                        inField[idx + xsize];        // j+1
    
    __syncthreads();
    
    // Second laplacian
    double laplap = -4.0 * tmp1Field[tmp_idx] + 
                    tmp1Field[tmp_idx - 1] +         // i-1
                    tmp1Field[tmp_idx + 1] +         // i+1
                    tmp1Field[tmp_idx - xsize] +     // j-1
                    tmp1Field[tmp_idx + xsize];      // j+1
    
    // Update field
    if (isLastIter) {
        outField[idx] = inField[idx] - alpha * laplap;
    } else {
        inField[idx] = inField[idx] - alpha * laplap;
    }
}
