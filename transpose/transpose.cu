#include <stdio.h>
#include <assert.h>
#include "helper_cuda.h"
#include "image.h"
#include "pgmIo.h"

#define STRING_MAX_SIZE 150
#define getElem(m, stride, i, j) m[(j) * stride + (i)]
#define TILE_DIM 16

void transposeCPU(Image * imgIn, Image *imgOut)
{
   for(int i = 0; i < imgOut->m; ++i )
     for (int j = 0; j < imgOut->n; ++j )
       getVal(imgOut, i, j) = getVal(imgIn, j, i);
}

__global__ void transposeNaive(const scalar * mIn, scalar * mOut, dim3 n)
{
}

__global__ void transposeShared(const scalar * mIn, scalar * mOut, dim3 n)
{
}

__global__ void transposeNoConflicts(const scalar * mIn, scalar * mOut, dim3 n)
{

}

#define NB_KERNELS 3

typedef void(*KernelPtr) (const scalar *, scalar *, dim3);

KernelPtr kernels[NB_KERNELS] = {&transposeNaive, &transposeShared, &transposeNoConflicts};
const char * kernelNames[NB_KERNELS]= {"naive", "shared", "noConflicts" };


void transposeGPU(Image const *imgIn, Image * imgOut, int indexKernel)
{
   scalar * mInDev,
          * mOutDev;

   dim3 tBlock(TILE_DIM, TILE_DIM);
   dim3 grid(imgIn->n / TILE_DIM, imgIn->m /TILE_DIM);

   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   size_t sizeTot = imgOut->n * imgOut->m  * sizeof(scalar);

   checkCudaErrors(cudaMalloc(&mInDev, sizeTot));
   checkCudaErrors(cudaMalloc(&mOutDev, sizeTot));
   checkCudaErrors(cudaMemcpy(mInDev, imgIn->matrix, sizeTot, cudaMemcpyHostToDevice));
   
   checkCudaErrors(cudaEventRecord(start, 0));

   dim3 dims(imgIn->m, imgIn->n);
   kernels[indexKernel]<<<grid, tBlock>>>(mInDev, mOutDev, dims);
   checkCudaErrors(cudaGetLastError());
     
   checkCudaErrors(cudaEventRecord(stop, 0));
   checkCudaErrors(cudaEventSynchronize(stop));
   float timeInMs;
   checkCudaErrors(cudaEventElapsedTime(&timeInMs, start, stop));
   printf("kernel %s : %f (ms)\n", kernelNames[indexKernel], timeInMs);
   checkCudaErrors(cudaMemcpy(imgOut->matrix, mOutDev, sizeTot, cudaMemcpyDeviceToHost));

   checkCudaErrors(cudaEventDestroy(start)); 
   checkCudaErrors(cudaEventDestroy(stop)); 

   checkCudaErrors(cudaFree(mInDev));
   checkCudaErrors(cudaFree(mOutDev));
}

int main(int argc, char * argv[])
{
   char filenameTrans[STRING_MAX_SIZE];
   Image img;
   Image * imgTrans,
         * imgTransCheck;

   const char * filename = (argc < 2) ? "data/cray.pgm" : argv[1];
   int indexKernel = (argc < 3) ? 0 : atoi(argv[2]);

   readPGM(filename, &img);

   if (img.n % TILE_DIM != 0) { fprintf(stderr, "image width is not multiple of %d\n", TILE_DIM); exit(-1); }
   if (img.m % TILE_DIM != 0) { fprintf(stderr, "image height is not multiple of %d\n", TILE_DIM); exit(-1); }

   imgTrans = allocImage(img.n, img.m, img.maxVal);
   imgTransCheck = allocImage(img.n, img.m, img.maxVal);
   
   transposeCPU(&img, imgTransCheck);
   transposeGPU(&img, imgTrans, indexKernel);
   addSuffixBeforeExt(filename, "Trans", filenameTrans); 
   writePGM(filenameTrans, imgTrans);

   // check phase
   for(int i = 0; i < imgTransCheck->m; ++i)
   {
     for(int j = 0; j < imgTransCheck->n; ++j)
     {
       if (fabs(static_cast<double>(getVal(imgTrans, i, j) - getVal(imgTransCheck, i, j))) > 1e-6)
       {
         fprintf(stderr, "erreur en %d %d\n", i, j);
         exit(-1);
       }       
     } 
   }
   freeImage(&img);
   freeImage(imgTrans);
   freeImage(imgTransCheck);
   return 0;
}
