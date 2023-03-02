#ifndef __MANDEL_CUDA_HPP
#define __MANDEL_CUDA_HPP

__global__ void mandelComputeCuda(uint32_t * buffer, float leftX, float topY, float xStep, float yStep, int WIN_DIM, int MAX_ITERATIONS);

#endif
