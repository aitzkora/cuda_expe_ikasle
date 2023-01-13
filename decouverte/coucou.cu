#include <stdio.h>

__global__ void pikaboo() {
    int i = threadIdx.x;
    printf("Peek-A-Boo from  %d  among %d threads\n", i, gridDim.x);
}

int main()  {
  pikaboo<<<1, 8>>>();
  cudaDeviceSynchronize();
  return 0;
}
