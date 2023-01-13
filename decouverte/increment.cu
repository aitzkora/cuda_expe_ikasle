#include <stdio.h>
#include <stdlib.h>
#define N 256

__global__ void increment(int * u, const int n, int val) {
  int i = threadIdx.x; u[i] += val;
}

int main()
{
  int tab[N];
  int * tab_d;
  int sizeCpy = N * sizeof(int);
  for(int i = 0; i < N; ++i)   tab[i] = 1;
  cudaMalloc(&tab_d, sizeCpy);
  cudaMemcpy(tab_d, tab, sizeCpy, cudaMemcpyHostToDevice );
  increment<<<1,N>>>(tab_d, N, 3);
  cudaMemcpy(tab, tab_d, sizeCpy, cudaMemcpyDeviceToHost );
  cudaFree(tab_d);
  for (int i = 0; i < N; ++i) { if (i[tab] != 4) abort();}
  return 0;
}
