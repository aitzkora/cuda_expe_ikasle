#ifndef __IMAGE_H__
#define __IMAGE_H__

#define getVal(pImg, i, j) (pImg)->matrix[(pImg)->n * (i) + (j)]

#include <stdint.h>

typedef uint32_t scalar;

struct Image {
  int m;
  int n;
  scalar maxVal;
  scalar * matrix;
};

// subroutine  to allocate a new Image
Image * allocImage(int m, int n, scalar maxVal);

void freeImage(Image * img);

#endif
