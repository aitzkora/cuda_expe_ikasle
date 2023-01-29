#include <stdio.h>
#include <math.h>
#include <helper_cuda.h>
#include <algorithm>

__global__ void kernel(float * a , int offset)
{
  float x, c, s;
  int i = offset + threadIdx.x+ blockIdx.x * blockDim.x;
  x = 1.0 * i;
  s = sin(x);
  c = cos(x);
  a[i] += sqrt(s*s + c*c);
}

int main()
{
  constexpr int nStreams = 4;
  const int blockSize = 256;
  const int n = 4 * 1024 * blockSize * nStreams;
  const size_t sizeTot = sizeof(float) * n;
  const int streamSize = n / nStreams;

  float * aPinned;
  float * aDevice;
  
  cudaStream_t streams[nStreams];

  checkCudaErrors(cudaMalloc(&aDevice, sizeTot));
  checkCudaErrors(cudaHostAlloc(&aPinned, sizeTot, cudaHostAllocDefault));

  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for(int i = 0; i < nStreams ; ++i)
     checkCudaErrors(cudaStreamCreate(&streams[i]));

  // sequential
  for(int i = 0; i < n ; ++i) { aPinned[i] = 0.0;}
  checkCudaErrors(cudaEventRecord(start, 0));

  checkCudaErrors(cudaMemcpy(aDevice, aPinned, sizeTot, cudaMemcpyHostToDevice));
  kernel<<<n/blockSize, blockSize>>>(aDevice, 0);
  checkCudaErrors(cudaMemcpy(aPinned, aDevice, sizeTot, cudaMemcpyDeviceToHost));

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));

  float time;
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
  float maxVal = 0.0;
  for(int i = 0; i < n ; ++i) { maxVal = std::max(fabsf(aPinned[i] -1.0), maxVal);}
  printf("Time for sequential %fms, error = %e\n", time, maxVal);
 
  // Asynchronous
  for(int i = 0; i < n ; ++i) { aPinned[i] = 0.0;}
  checkCudaErrors(cudaEventRecord(start, 0));

  for(int s = 0 ; s < nStreams; ++s) 
  {
    int offset = s * streamSize;
    checkCudaErrors(cudaMemcpyAsync(aDevice + offset, aPinned + offset, streamSize * sizeof(float), 
                    cudaMemcpyHostToDevice, streams[s]));
    kernel<<<streamSize/blockSize, blockSize, 0, streams[s]>>>(aDevice, offset);
    checkCudaErrors(cudaMemcpyAsync(aPinned + offset, aDevice + offset, streamSize * sizeof(float), 
                    cudaMemcpyDeviceToHost, streams[s]));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&time, start, stop));
  
  maxVal = 0.0;
  for(int i = 0; i < n ; ++i) { maxVal = std::max(fabsf(aPinned[i] -1.0), maxVal);}
  printf("Time for asynchronous %fms, error = %e\n", time, maxVal);
 
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFreeHost(aPinned);

}
