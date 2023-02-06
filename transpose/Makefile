.SUFFIXES:.cu .x

SM_NUM ?= 	75
CUDA_PATH ?= /opt/cuda
CUDA=nvcc
CFLAGS= -I$(NVHPC)/examples/OpenACC/SDK/include  -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets

all:: transpose.x 

transpose.x: transpose.cu image.cu image.h pgmIo.cu pgmIo.h
	$(CUDA) $(CFLAGS) -o $@ transpose.cu image.cu pgmIo.cu 
clean:
	@echo  Cleaning...
	rm -f *.x
	rm -f data/*Trans.pgm
