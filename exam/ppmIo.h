#ifndef __PPMIO_H__
#define __PPMIO_H__

#include "image.h" 

void writePPM(const char * filename, const Image * pImg, short verbose);

#endif
