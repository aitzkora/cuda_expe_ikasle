#include <stdio.h> 
#include "image.h"

Image * allocImage(int m, int n, scalar maxVal)
{
  Image * pImg;

  pImg = (Image * )malloc(sizeof(Image*));
  if (!pImg)
  {
    fprintf(stderr, "cannot alloc a new Image!\n");
    exit (-1);
  }
  pImg->m = m;
  pImg->n = n;
  pImg->maxVal = maxVal;

  pImg->matrix = (scalar *) malloc(sizeof(scalar) * m * n);

  if (!pImg->matrix)
  {
    fprintf(stderr, "cannot alloc a new Image of size (%d,%d) !\n", m, n);
    exit (-1);
  }  
  return pImg;
}

void freeImage(Image * img)
{
   free(img->matrix);
}

