// From a idea of PAP
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "helper_cuda.h"
#include <math.h>

__global__ void scale(float * vec, float k, int size)
{
}

__global__ void scaleFlip(float * vec, float k, int size)
{
}

__global__ void scaleFlipAndHalf(float * vec, float k, int size)
{
}

void cpuScale(float * vec, float k, int size)
{
  for(int i = 0; i < size; ++i)  vec[i] *= k; 
}

int main(int argc, char * argv[])
{

   int N = (argc < 2 ) ? 4*1024*1024 : atoi(argv[1]);
   int blockSize  = (argc < 3) ? 256 : atoi(argv[2]);
   int sizeTot = N * sizeof(float);

   float * x,
         * x2,
         * xCheck;

   if ((x = (float *) malloc(sizeTot)) == NULL) {
      printf("cannot allocate x\n"); abort();
   }

   if ((x2 = (float *) malloc(sizeTot)) == NULL) {
      printf("cannot allocate x2\n"); abort();
   }

   if ((xCheck = (float *) malloc(sizeTot)) == NULL) {
      printf("cannot allocate xCheck\n"); abort();
   }


   if (N % blockSize != 0)
   {
     fprintf(stderr, "%d is not a multiple of %d!\n", N, blockSize);
     exit (-1);
   }


   for(int i = 0; i < N; ++i) {x[i]=1. * i;}

   float *xDevice;
 
   // struct of arrays allocation and initialization

   cudaMalloc(&xDevice, sizeTot);
   checkCudaErrors(cudaMemcpy(xDevice, x, sizeTot, cudaMemcpyHostToDevice ));

   bool sizeInMB = N > (1<<20);
   const char * unitStr = sizeInMB ? "MB" : "B";
   printf("data size (%s) : %d\n", unitStr, (sizeInMB ? (N>>20) : N ));
   printf("blocksize (B) : %d\n", blockSize);

   float pi = 4.0 * atanf(1.0);

   // to check that kernels give good results
   
   for(int i = 0; i < N; ++i) x2[i]=x[i];

   cpuScale (x, pi, N);

   scale<<<N/blockSize, blockSize>>>(xDevice, pi, N);

   checkCudaErrors(cudaMemcpy(xCheck, xDevice, sizeTot, cudaMemcpyDeviceToHost ));

   for(int i = 0; i < N; ++i) { assert (fabs(xCheck[i] - x[i])<1e-6); }

   checkCudaErrors(cudaMemcpy(xDevice, x2, sizeTot, cudaMemcpyHostToDevice ));

   scaleFlipAndHalf<<<N/blockSize, blockSize>>>(xDevice, pi, N);
   
   checkCudaErrors(cudaMemcpy(x2, xDevice, sizeTot, cudaMemcpyDeviceToHost ));
   
   for(int i = 0; i < N; ++i) { assert (fabs(xCheck[i] - x2[i])<1e-6); }

   scale<<<N/blockSize, blockSize>>>(xDevice, pi, N);
   scaleFlip<<<N/blockSize, blockSize>>>(xDevice, pi, N);
   scaleFlipAndHalf<<<N/blockSize, blockSize>>>(xDevice, pi, N);

   // memoria askatu
   cudaFree(xDevice);
   free(x2);
   free(xCheck);
   free(x);
   return 0;
}
