#ifndef __IMAGE_H__
#define __IMAGE_H__

#define getVal(pImg, i, j) (pImg)->matrix[(pImg)->n * (i) + (j)]

typedef int scalar;

struct Image {
  int m;
  int n;
  scalar maxVal;
  scalar * matrix;
};

// subroutine  to allocate a new Image
Image * allocImage(int m, int n, int maxVal);

void freeImage(Image * img);

#endif
