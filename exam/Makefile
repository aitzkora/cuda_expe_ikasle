.SUFFIXES:.cu .x

SM_NUM ?= 	75
CUDA_PATH ?= /opt/cuda
CUDA=nvcc
CFLAGS= -I$(NVHPC)/examples/OpenACC/SDK/include  -arch=sm_$(SM_NUM) 

all:: mandel_seq.x  mandel_cuda.x

mandel_seq.x : mandel_seq.cu mandelbrot_seq.cu ppmIo.h image.h mandel_funs.h mandel_seq.h mandel_funs.cu
	$(CUDA) $(CFLAGS) -o $@ mandel_seq.cu mandelbrot_seq.cu image.cu ppmIo.cu mandel_funs.cu

mandel_cuda.x : mandel_cuda.cu mandelbrot_cuda.cu ppmIo.h image.h mandel_funs.h mandel_cuda.h mandel_funs.cu 
	$(CUDA) $(CFLAGS) -o $@ mandel_cuda.cu mandelbrot_cuda.cu image.cu ppmIo.cu mandel_funs.cu 

clean_all :: clean clean_data
clean:
	@echo  Cleaning...
	rm -f *.x

clean_data:
	@echo cleaning data files
	rm -fr film
	rm -f output.mkv
