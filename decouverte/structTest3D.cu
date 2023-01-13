// From a idea of PAP
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include "helper_cuda.h"
#include <vector>

struct AOS{
   float x, y, z;
   float dx, dy, dz;
};

struct SOA
{
  float * x, *y, *z;
  float * dx, *dy, *dz;
};


__global__ void aosKernel(AOS * pos, int size)
{
}

__global__ void soaKernel(SOA pos, int size)
{
}

int main(int argc, char * argv[])
{
   float time;
   cudaEvent_t startEvent, stopEvent;
   const int nMB = 1024 * 1024;
   // arg 1 : size in MB
   int N = (argc < 2 ) ? 8 * nMB : atoi(argv[1]) * nMB;
   // arg 2 : blockSize
   int blockSize  = (argc < 3) ? 512 : atoi(argv[2]);
   int numDevice  = (argc < 4) ? 0 : atoi(argv[3]);

   const int sizeTotAOS = N * sizeof(AOS);     
   const int sizeTotSOA = N * sizeof(float);

   float *x;
   float *y;
   float *z;

   float *dx;
   float *dy;
   float *dz;

   x = (float *) malloc(N*sizeof(float));
   y = (float *) malloc(N*sizeof(float));
   z = (float *) malloc(N*sizeof(float));
   dx = (float *) malloc(N*sizeof(float));
   dy = (float *) malloc(N*sizeof(float));
   dz = (float *) malloc(N*sizeof(float));


   SOA structOfArray;
   AOS * arrayOfStruct, 
       * arrayOfStructDevice;

   cudaEventCreate(&startEvent, 0);
   cudaEventCreate(&stopEvent, 0);

   // allocation
   checkCudaErrors(cudaMalloc(&structOfArray.x, sizeTotSOA));
   checkCudaErrors(cudaMalloc(&structOfArray.dx, sizeTotSOA));
   checkCudaErrors(cudaMalloc(&structOfArray.y, sizeTotSOA));
   checkCudaErrors(cudaMalloc(&structOfArray.dy, sizeTotSOA));
   checkCudaErrors(cudaMalloc(&structOfArray.z, sizeTotSOA));
   checkCudaErrors(cudaMalloc(&structOfArray.dz, sizeTotSOA));

   for(int i = 0; i < N ; i ++)   x[i] = 1.0;
   for(int i = 0; i < N ; i ++)   y[i] = 2.0;
   for(int i = 0; i < N ; i ++)   z[i] = 3.0;
   for(int i = 0; i < N ; i ++)  dx[i] = -1.0;
   for(int i = 0; i < N ; i ++)  dy[i] =  2.0;
   for(int i = 0; i < N ; i ++)  dz[i] = -3.0;

   checkCudaErrors(cudaMemcpy(structOfArray.x, x ,sizeTotSOA, cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy(structOfArray.dx, dx, sizeTotSOA, cudaMemcpyHostToDevice ));

   checkCudaErrors(cudaMemcpy(structOfArray.y, y, sizeTotSOA, cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy(structOfArray.dy, dy, sizeTotSOA, cudaMemcpyHostToDevice ));

   checkCudaErrors(cudaMemcpy(structOfArray.z, z, sizeTotSOA, cudaMemcpyHostToDevice ));
   checkCudaErrors(cudaMemcpy(structOfArray.dz, dz, sizeTotSOA, cudaMemcpyHostToDevice ));

   arrayOfStruct= (AOS *) malloc(sizeTotAOS);

   for(int i = 0; i < N; ++i)
   {
     arrayOfStruct[i].x = x[i];
     arrayOfStruct[i].dx = dx[i];
     arrayOfStruct[i].y = y[i];
     arrayOfStruct[i].dy = dy[i];
     arrayOfStruct[i].z = z[i];
     arrayOfStruct[i].dz = dz[i];
   }

   checkCudaErrors(cudaMalloc(&arrayOfStructDevice, sizeTotAOS));
   checkCudaErrors(cudaMemcpy(arrayOfStructDevice, arrayOfStruct, sizeTotAOS, cudaMemcpyHostToDevice));

   bool sizeInMB = N > (1<<20);
   const char * unitStr = sizeInMB ? "MB" : "B";
   cudaSetDevice(numDevice);
   int devNum;
   cudaGetDevice(&devNum);

   printf("Device : %d \n", devNum);
   printf("data size (%s) : %d\n", unitStr, (sizeInMB ? (N>>20) : N ));
   printf("blocksize (B) : %d\n", blockSize);

   soaKernel<<<(N-1)/blockSize + 1, blockSize>>>(structOfArray, N);

   aosKernel<<<(N-1)/blockSize + 1, blockSize>>>(arrayOfStructDevice, N);

   printf("aos Kernel time : %04.4fms\n", time);
 
   checkCudaErrors(cudaEventDestroy(startEvent));
   checkCudaErrors(cudaEventDestroy(stopEvent));

   // memoria askatu
   checkCudaErrors(cudaFree(arrayOfStructDevice));
   checkCudaErrors(cudaFree(structOfArray.x));
   checkCudaErrors(cudaFree(structOfArray.dx));
   checkCudaErrors(cudaFree(structOfArray.y));
   checkCudaErrors(cudaFree(structOfArray.dy));
   checkCudaErrors(cudaFree(structOfArray.z));
   checkCudaErrors(cudaFree(structOfArray.dz));
   // liberation de la memoire sur l'hote

   free(x);
   free(y);
   free(z);
   free(dx);
   free(dy);
   free(dz);

   return 0;
}
