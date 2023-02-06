#ifndef __PGMIO_H__
#define __PGMIO_H__

#include "image.h" 

void readPGM(const char * filename, Image * img);
void writePGM(const char * filename, const Image * pImg);
void addSuffixBeforeExt(const char * filename, const char * suffix, char * filenameWithSuffix);

#endif
