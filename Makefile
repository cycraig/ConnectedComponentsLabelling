CFLAGS=-O3 -g
NVCCFLAGS= -Wno-deprecated-gpu-targets
NVCCFLAGS += -Xcompiler -fopenmp
LDFLAGS= -lglut -lGL -lGLU -lgomp
SHAREDH=./src/common_ccl.h
SHAREDC=./src/EasyBMP.cpp ./src/common_ccl.cpp
MPIFLAGS=-DOMPI_SKIP_MPICXX
CC=nvcc
MPICC=mpiCC

all: ccl_unionfind ccl_gpu ccl_mpi ccl_gpu_global

#ccl: ./src/ccl.cu $(SHAREDH) $(SHAREDC)
#	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl.cu $(LDFLAGS) -o ccl

#ccl_fast: ./src/ccl_fast.cu $(SHAREDH) $(SHAREDC)
#	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl_fast.cu $(LDFLAGS) -o ccl_fast

ccl_unionfind: ./src/ccl_unionfind.cpp $(SHAREDH) $(SHAREDC)
	$(MPICC) $(CFLAGS) $(SHAREDC) $(MPIFLAGS) ./src/ccl_unionfind.cpp $(LDFLAGS) -o ccl_unionfind

ccl_gpu: ./src/ccl_gpu.cu $(SHAREDH) $(SHAREDC)
	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl_gpu.cu $(LDFLAGS) -arch compute_30 -o ccl_gpu

ccl_gpu_global: ./src/ccl_gpu_global.cu $(SHAREDH) $(SHAREDC)
	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl_gpu_global.cu $(LDFLAGS) -arch compute_30 -o ccl_gpu_global

ccl_mpi: ./src/ccl_mpi.cpp $(SHAREDH) $(SHAREDC)
	$(MPICC) $(CFLAGS) $(SHAREDC) $(MPIFLAGS) ./src/ccl_mpi.cpp $(LDFLAGS) -o ccl_mpi

clean:
	rm ccl_unionfind ccl_gpu ccl_mpi ccl_gpu_global
