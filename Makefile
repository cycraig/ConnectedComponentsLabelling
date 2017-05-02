CFLAGS=-O1 -g 
NVCCFLAGS= -Wno-deprecated-gpu-targets
#NVCCFLAGS += -Xcompiler -fopenmp
LDFLAGS= -lglut -lGL -lGLU -lgomp
SHAREDH=./src/common_ccl.h
SHAREDC=./src/EasyBMP.cpp ./src/common_ccl.cpp
MPIFLAGS=-DOMPI_SKIP_MPICXX
CC=nvcc

all: ccl ccl_fast ccl_gpu ccl_mpi

ccl: ./src/ccl.cu $(SHAREDH)
	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl.cu $(LDFLAGS) -o ccl

ccl_fast: ./src/ccl_fast.cu $(SHAREDH)
	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl_fast.cu $(LDFLAGS) -o ccl_fast

ccl_gpu: ./src/ccl_gpu.cu $(SHAREDH)
	$(CC) $(CFLAGS) $(NVCCFLAGS) $(SHAREDC) ./src/ccl_gpu.cu $(LDFLAGS) -arch compute_30 -o ccl_gpu

ccl_mpi: ./src/ccl_mpi.cpp $(SHAREDH)
	mpiCC $(CFLAGS) $(SHAREDC) $(MPIFLAGS) ./src/ccl_mpi.cpp $(LDFLAGS) -o ccl_mpi

clean:
	rm ccl ccl_fast ccl_gpu ccl_mpi
