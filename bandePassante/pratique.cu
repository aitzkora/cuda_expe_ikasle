#include <cstdio> 

#define N 8*1024*1024
#define BLOCK_SIZE 256

__global__ void base(float * a, float *b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = exp(1.+sin(b[i]));
}

__global__ void memory(float * a, float *b)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = b[i];
}

__global__ void math(float * a, float b, int flag)
{
  float v;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  v = exp(1.+sin(b+b*b));
  if (v*flag == 1) a[i] = v;
}

int main()
{
  size_t sizeTot = N * sizeof(float);

  float * aDevice,
        * bDevice;

  cudaMalloc(&aDevice, sizeTot);
  cudaMalloc(&bDevice, sizeTot);

  float * a = (float*)malloc(sizeTot);
  float * b = (float*)malloc(sizeTot);

  for(int i = 0; i < N ; ++i) { b[i] = 1.;}

  cudaMemcpy(bDevice, b, sizeTot, cudaMemcpyHostToDevice);

  base<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(aDevice, bDevice);
  memory<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(aDevice, bDevice);
  math<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(aDevice, 1.0, 0);

  cudaMemcpy(a, aDevice, sizeTot, cudaMemcpyDeviceToHost);

  printf("%d\n", a[1]);

  cudaFree(a);
  cudaFree(b);
  free(a);
  free(b);
  return 0;
}
