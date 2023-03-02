#include "mandel_funs.h"
#include <stdint.h>

static 
// ajouter un qualifieur CUDA ici pour pouvoir appeler cette fonction depuis le noyau mandelComputeCuda
.....
uint32_t iteration_to_color_dev(int iter) {
  uint32_t r, g, b;
  r = 0;
  g = 0;
  b = 0;
  if (iter < 63) 
    r = iter * 2;
  else if (iter < 127)
    r = (((iter - 64) * 128) / 126) + 128;
  else if (iter < 256)
    r = (((iter - 128) * 62) / 127) + 193;
  else if (iter < 512) {
    r = 255;
    g = (((iter - 256) * 62) / 255) + 1;
  }
  else if (iter <= 1024) {
    r = 255;
    g = (((iter - 512) * 63) / 511) + 64;
  }
  else if (iter <= 2048) {
    r = 255;
    g = (((iter - 1024) * 63) / 1023) + 128 ;
  } 
  else {
    r = 255;
    g = (((iter - 2048) * 63) / 2047) + 192; 
  }  
  return r << 24 | g << 16 | b << 8| 255;
}

__global__ void  mandelComputeCuda(uint32_t * buffer, float leftX, float topY, float xStep, float yStep, int WIN_DIM, int MAX_ITERATIONS)
{ 

} 
