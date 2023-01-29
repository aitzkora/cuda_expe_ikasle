.SUFFIXES:

SM_NUM ?= 75

CUDA=nvcc
CFLAGS= -O2 -arch=sm_$(SM_NUM) -allow-unsupported-compiler  # --ptxas-options=-v

all: pratique.x pratiqueFM.x theo.x 

pratiqueFM.x: pratique.cu
	$(CUDA) $(CFLAGS) --use_fast_math -o $@ $<

pratique.x: pratique.cu
	$(CUDA) $(CFLAGS) -o $@ $<

theo.x: theo.cu
	$(CUDA) $(CFLAGS) -o $@ $<

clean: 
	rm -fr *.x
