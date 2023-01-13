.SUFFIXES:.cu .x

SM_NUM ?= 75
CUDA_PATH ?= /opt/cuda

CUDA=nvcc
CFLAGS = -allow-unsupported-compiler -O2 -I$(NVHPC)/examples/OpenACC/SDK/include -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets
#CFLAGS = -I$(NVHPC)/examples/OpenACC/SDK/include -arch=sm_$(SM_NUM) -Wno-deprecated-gpu-targets

all:: coucou.x scale.x increment.x structTest3D.x

.cu.x:
	$(CUDA) $(CFLAGS) -o $@ $<

clean:
	@echo  Cleaning...
	rm -f *.x
