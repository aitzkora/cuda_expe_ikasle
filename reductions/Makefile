.SUFFIXES: .cu .x

SM_NUM ?= 75
CUDA_PATH ?= /opt/cuda

CUDA=nvcc
CFLAGS= -I$(NVHPC)/examples/OpenACC/SDK/include  -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets  #--ptxas-options=-v
LDFLAGS=-L$(CUDA_PATH)/lib64 -lcurand
all:  monteCarlo.x

.cu.x: 
	$(CUDA) $(CFLAGS) $(LDFLAGS) -o $@ $< 

clean: 
	rm -fr *.x
