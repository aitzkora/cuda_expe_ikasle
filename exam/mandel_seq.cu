#include "mandel_funs.h"
#include <stdint.h>

static uint32_t iterationToColor(int iter) {
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

void mandelComputeSeq(uint32_t * buffer, float leftX, float topY, float xStep, float yStep, int MAX_ITERATIONS)
{
    int  i,j;
    for(i=0; i < WIN_DIM; i++) {
      for (j=0; j <WIN_DIM; j++) {
        int iter;
        float cr, ci, zr, zi, x2, y2, twoXY;

        cr = leftX + xStep * j;
        ci = topY - yStep * i;
        zr = 0.;
        zi = 0.;
        for(iter = 0; iter < MAX_ITERATIONS; iter ++) { 
          x2 = zr * zr;
          y2 = zi * zi;
          if (x2 + y2 > 4.0) 
            break;
          twoXY = 2.0 * zr * zi;
          zr = x2 - y2 + cr;
          zi = twoXY + ci;
        }  
        buffer[i * WIN_DIM + j] = 
            (iter == MAX_ITERATIONS) ? 255 : iterationToColor(iter);
      }
    }
}
