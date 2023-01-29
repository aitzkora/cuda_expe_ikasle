#include <cuda_runtime.h>
#include <cstdio>

int main(int argc, char * argv[])
{
  int numDevice = (argc < 2 ) ? 0 : atoi(argv[1]);
 
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, numDevice);

  int clockRate = prop.memoryClockRate;
  int busWidth = prop.memoryBusWidth;

  printf("Nom de l'accelerateur : %s\n", prop.name);
  printf("Fréquence (Khz) : %d\n", clockRate);
  printf("Largeur de bus (bits) : %d\n", busWidth);
  printf("Pic de bande passante(Go/s) : %6.2f\n", 2.0 * clockRate * (busWidth / 8) * 1.e-6);

  return 0;
}
