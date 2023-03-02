#include "image.h"
#include "mandel_cuda.h"
#include "mandel_funs.h"
#include <math.h>
#include <stdio.h>
#include "helper_cuda.h"
#include "ppmIo.h"

float ZOOM_SPEED = -0.01;
float leftX     = -0.2395;
float rightX    = -0.2275;
float topY      =  0.660;
float bottomY   =  0.648;
float xStep, yStep;

int WIN_DIM, MAX_ITERATIONS;

int main(int argc, char * argv[]) {
  Image * img;
  
  int OUTER_ITERATIONS;
  WIN_DIM = (argc >= 2 ) ? atoi(argv[1]) : 200 ;
  OUTER_ITERATIONS = (argc >= 3) ? atoi(argv[2]) : 100;


  char filename[1024];
  mandelInit(WIN_DIM, 256);

  img = allocImage(WIN_DIM, WIN_DIM, (uint32_t) 0xFF);
  size_t sizeBuffer = WIN_DIM * WIN_DIM * sizeof(uint32_t);
  uint32_t * buffer = img->matrix;

  // create directory for images
  system("mkdir -p film");

  // declare a buffer for the device
  uint32_t * buffer_d;

  // and allocate it on the GPU
  ... 

  // set grid parameters
  dim3 blockSize(32, 32);
  ...

  short verbose=0;

  // display progress bar
  printf("<-----------------------------------------------");
  printf("---------------------------------------------------|\r"); 


  // main loop
  for(int i = 0; i < OUTER_ITERATIONS; i++) {
    // appel du noyau
    mandelComputeCuda<<<..., blockSize>>>(buffer_d, leftX, topY, xStep, yStep, WIN_DIM, MAX_ITERATIONS);
    /// copie depuis le GPU vers l'hote
     .... 

    // ecriture sur disque
    snprintf(filename, 1023, "film/mandel_cuda_%03d.ppm", i);
    writePPM(filename, img, verbose);
    zoom();
    if( i * 100 % OUTER_ITERATIONS  == 0) {
       printf("\b->");
       fflush(stdout);
    }
  }
  //system("ffmpeg -framerate 25 -i 'film/mandel_%03d.ppm' -y output.mkv");
  
  //free buffer
  ....

  // free img
  freeImage(img);
}
