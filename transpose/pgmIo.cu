#include <stdio.h>
#include "image.h"

// non robust routine for reading pgm file (P2 version)
void readPGM(const char * filename, Image * pImg)
{
  FILE * fp;
  fp = fopen(filename, "r");
  if (!fp)
  {
    fprintf(stderr, "can't open %s file \n", filename);
    exit (-1);
  }
  char magic[3]; 

  int maxValRead;
  fscanf(fp, "%s\n", magic);
  fscanf(fp, "%d %d", &pImg->m, &pImg->n);
  fscanf(fp, "%d", &maxValRead);
  pImg->maxVal = static_cast<scalar>(maxValRead);
  printf("Reading image in the file %s : it has %d %d lines and columns\n", filename, pImg->m, pImg->n);
  pImg->matrix = (scalar *) malloc(sizeof(scalar)*pImg->m*pImg->n);
   
  for(int j = 0; j < pImg->n ; ++j) 
  {
    for(int i = 0; i < pImg->m ; ++i)
    {
      int tmp;
      fscanf(fp, "%d", &tmp);
      getVal(pImg, i, j) = static_cast<scalar>(tmp);
    }
  }
  fclose(fp);
}

void writePGM(const char * filename, const Image * pImg)
{
  FILE * fp;
  fp = fopen(filename, "w");
  const char * magic = "P2";

  if (!fp)
    {
      fprintf(stderr, "can't open %s file \n", filename);
      exit (-1);
    }
  fprintf(fp, "%s\n", magic);
  fprintf(fp, "%d %d\n", pImg->m, pImg->n);
  fprintf(fp, "%d\n", (int) pImg->maxVal);
  printf("Writing an image %dx%d in the file %s\n", pImg->m, pImg->n, filename);
  for(int j = 0; j < pImg->n ; ++j)
  {
    int count = 0;
    for(int i = 0; i < pImg->m ; ++i)
    {
      count +=4; 
      if (count < 70) 
      {
        fprintf(fp, "%03d ", (int) getVal(pImg, i, j) );
      }
      else
      {
         count = 4;
         fprintf(fp,"\n%03d ", (int) getVal(pImg, i, j) );
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
