.SUFFIXES: .cu .x

SM_NUM ?= 75
CUDA_PATH ?= /opt/cuda

CUDA=nvcc
CFLAGS=-I$(NVHPC)/examples/OpenACC/SDK/include/  -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets -allow-unsupported-compiler  #--ptxas-options=-v

all: pinned.x exAsync.x

.cu.x: 
	$(CUDA) $(CFLAGS) -o $@ $<

clean: 
	rm -fr *.x
