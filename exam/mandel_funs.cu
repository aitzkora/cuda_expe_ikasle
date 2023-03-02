#include "mandel_funs.h"

void mandelInit(int winDim, int maxIt) {
    WIN_DIM = winDim;
    MAX_ITERATIONS = maxIt;
    xStep = (rightX - leftX) / WIN_DIM;
    yStep = (topY - bottomY) / WIN_DIM;
} 
  
void zoom() {
    float xRange, yRange ;

    xRange = (rightX - leftX);
    yRange = (topY - bottomY);

    leftX = leftX + ZOOM_SPEED * xRange;
    rightX = rightX - ZOOM_SPEED * xRange;
    topY = topY - ZOOM_SPEED * yRange;
    bottomY = bottomY + ZOOM_SPEED * yRange;

    xStep = (rightX - leftX) / WIN_DIM;
    yStep = (topY - bottomY) / WIN_DIM;
}

