#ifndef __MANDEL_FUNS_H_
#define __MANDEL_FUNS_H_

extern  float ZOOM_SPEED;
extern  float leftX     ;
extern  float rightX    ;
extern  float topY      ;
extern  float bottomY   ;
extern  float xStep, yStep;
extern  int WIN_DIM, MAX_ITERATIONS;


void zoom();
void mandelInit(int winDim, int maxIt);


#endif
