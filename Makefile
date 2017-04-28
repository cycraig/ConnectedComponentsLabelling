CFLAGS=-O1 -g -Wno-deprecated-gpu-targets
INC=-I./inc
LDFLAGS= -lglut -lGL -lGLU
SHAREDC=./src/EasyBMP.cpp
CC=nvcc

all: ccl ccl_fast ccl_gpu

ccl: ./src/ccl.cu
	$(CC) $(CFLAGS) $(SHAREDC) ./src/ccl.cu $(INC) $(LDFLAGS) -o ccl

ccl_fast: ./src/ccl_fast.cu
	$(CC) $(CFLAGS) $(SHAREDC) ./src/ccl_fast.cu $(INC) $(LDFLAGS) -o ccl_fast

ccl_gpu: ./src/ccl_gpu.cu
	$(CC) $(CFLAGS) $(SHAREDC) ./src/ccl_gpu.cu $(INC) $(LDFLAGS) -arch compute_30 -o ccl_gpu

clean:
	rm ccl ccl_fast ccl_gpu
