#include <stdio.h>
#include "image.h"
#include "ppmIo.h"

void writePPM(const char * filename, const Image * pImg, short verbose)
{
  FILE * fp;
  fp = fopen(filename, "w");
  const char * magic = "P3";

  if (!fp)
    {
      fprintf(stderr, "can't open %s file \n", filename);
      exit (-1);
    }
  fprintf(fp, "%s\n", magic);
  fprintf(fp, "%d %d\n", pImg->m, pImg->n);
  fprintf(fp, "%d\n", (int) pImg->maxVal);
  if (verbose)
    printf("Writing an image %dx%d in the file %s\n", pImg->m, pImg->n, filename);
  for(int j = 0; j < pImg->n ; ++j)
  {
    int count = 0;
    for(int i = 0; i < pImg->m ; ++i)
    {
      scalar valCurr = (scalar) getVal(pImg, i, j);
      unsigned char r = valCurr >> 24;
      unsigned char g = (valCurr >> 16) && 255;
      unsigned char b = (valCurr >> 8) && 255;
      count +=12; 
      if (count < 70) 
      {
        fprintf(fp, "%03d %03d %03d ", r, g, b );
      }
      else
      {
        count = 12;
        fprintf(fp,"\n%03d %03d %03d ", r, g, b );
      }
    }
    fprintf(fp, "\n");
  }
  fclose(fp);
}


void addSuffixBeforeExt(const char * filename, const char * suffix, char * filenameWithSuffix)
{
  size_t pos = (size_t)(strchr(filename, '.') - filename);
  strncpy(filenameWithSuffix, filename, pos);
  filenameWithSuffix[pos] = '\0';
  strcat(filenameWithSuffix, suffix);
  strcat(filenameWithSuffix, ".pgm");
}
