#include <stdio.h>

#define N 16*1024*1024
typedef float scalar;

int main()
{
   scalar * pinnedTab;
   scalar * tabDevice;
   scalar * tab;
   size_t sizeTot = N * sizeof(scalar);

   cudaHostAlloc(&pinnedTab, sizeTot, cudaHostAllocDefault);
   cudaMalloc(&tabDevice, sizeTot);
  
   tab = (scalar*) malloc(sizeTot);

   for(int i = 0; i < N ; ++i) { pinnedTab[i] = (scalar)(1.0) * i; }
   for(int i = 0; i < N ; ++i) { tab[i] = (scalar)(1.0) * i; }

   cudaMemcpy(tabDevice, pinnedTab, sizeTot,  cudaMemcpyHostToDevice);

   cudaMemcpy(tabDevice, tab, sizeTot, cudaMemcpyHostToDevice);

   cudaFree(tabDevice);
   cudaFreeHost(pinnedTab);
   free(tab);
}
