// adapted from CUDA fortran for Scientists and Engineers
#include <cstdio>
#include "helper_cuda.h"
#include <vector>
#include "curand.h"
typedef float scalar;


__global__ void finalSum(int * partial, int * total)
{
  extern __shared__ int psum[];
  int i = threadIdx.x;
  psum[i] = partial[i];
  __syncthreads();
  int  iNext = blockDim.x/2;
  while (iNext > 0)
  {
    if (i < iNext)
      psum[i] += psum[i+iNext];
    iNext >>= 1;
    __syncthreads();
  }
  if (i == 0) *total = psum[0];
}

__global__ void partialSum(scalar * input, int * partial, const int N) 
{
  extern __shared__ int psum[];
  int idX  = threadIdx.x + blockDim.x * blockIdx.x;
  int interior = 0;
  for (int i = idX; i < N ; i+= gridDim.x * blockDim.x)
    if ((input[i]*input[i]+input[i+N]*input[i+N]) <= 1.0)
       interior++;
  idX = threadIdx.x;
  psum[idX] = interior;
  __syncthreads();
  int iNext = blockDim.x / 2;
  while (iNext > 0)
  {
    if (idX < iNext)
      psum[idX] += psum[idX+iNext];
    iNext >>= 1;
    __syncthreads();
  }
  if (idX == 0) partial[blockIdx.x] = psum[0];
}

__global__ void partialSumDiverge(scalar *input, int * partial, const int N)
{
}

__global__ void partialSumDummy(scalar *input, int * partial, const int N)
{
}

scalar computeSum(const scalar * XY, int N)
{
  int interior = 0;
  for(int i = 0; i < N; ++i)
  {
    if ((XY[i] * XY[i] + XY[i+N]*XY[i+N]) <= static_cast<scalar>(1.0))
       interior++;
  }
  return interior/static_cast<scalar>(N);
}

#define BLOCK_SIZE 512
#define NB_BLOCKS 256
#define NB_KERNELS 3
typedef void(*KernelPtr) (scalar *, int * partial, const int N);
KernelPtr kernels[NB_KERNELS] = {&partialSum, &partialSumDiverge, partialSumDummy};
const char * kernelNames[NB_KERNELS]= {"noDiverging", "diverging", "oneThread" };

int main(int argc, char * argv[])
{
  const int N = (argc < 2) ? 4 * NB_BLOCKS * BLOCK_SIZE  : atoi(argv[1]);
  const int twoN = N << 1;

  int * partial;
  cudaMalloc(&partial, NB_BLOCKS * sizeof(int));
  int * interiorGPU;
  cudaHostAlloc(&interiorGPU, sizeof(int), cudaHostAllocMapped);
  // xy data on host;
  scalar * xy = (scalar*) malloc(twoN * sizeof(scalar));
  


  // xy data on Device
  scalar * xyDevice;
  size_t sizeXY = twoN * sizeof(scalar);
  checkCudaErrors(cudaMalloc(&xyDevice, sizeXY));

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  int seed  = 1234;
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniform(gen, xyDevice, twoN);

  checkCudaErrors(cudaMemcpy(xy, xyDevice, sizeXY, cudaMemcpyDeviceToHost));

  scalar res = computeSum(xy, N);

  partialSum<<<NB_BLOCKS,BLOCK_SIZE,BLOCK_SIZE*sizeof(int)>>>(xyDevice, partial, N);
  checkCudaErrors(cudaGetLastError());
  finalSum<<<1,NB_BLOCKS,NB_BLOCKS*sizeof(int)>>>(partial, interiorGPU);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  scalar resGPU = *interiorGPU/static_cast<scalar>(N);
    if (fabsf(resGPU-res) > 1.e-6)
  {
    fprintf(stderr, "********* ERROR ********** \n bad kernel computation\n");
  }
  else
  { 
    printf("********* SUCCESS *********** \nN = %d, π ~ %f\n", N ,4.0 * resGPU);
  }

  // chrono des reductions
  
  //for(int ker = 0; ker < 3; ++ker) decommenter ici quand les autres noyaux sont implementes
  for(int ker = 0; ker < 1; ++ker) 
  {
    cudaEvent_t start, stop;
    float timeInMs;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    checkCudaErrors(cudaEventRecord(start, 0));
    int nbTries = 10;
    for(int j = 0; j < nbTries ; ++j) {
       kernels[ker]<<<NB_BLOCKS,BLOCK_SIZE,BLOCK_SIZE*sizeof(int)>>>(xyDevice, partial, N);
    }
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&timeInMs, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    printf("ker = %s, time = %f\n", kernelNames[ker], timeInMs / nbTries);
  } 

  checkCudaErrors(cudaFree(xyDevice));
  checkCudaErrors(cudaFree(partial));
  checkCudaErrors(cudaFreeHost(interiorGPU));
  free(xy);
  return 0;
}
