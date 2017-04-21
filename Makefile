CFLAGS=-O1 -g -Wno-deprecated-gpu-targets
INC=-I./inc
LDFLAGS= -lglut -lGL -lGLU
SHAREDC=./src/EasyBMP.cpp
CC=nvcc

all: ccl ccl_fast

ccl: ./src/ccl.cu
	$(CC) $(CFLAGS) $(SHAREDC) ./src/ccl.cu $(INC) $(LDFLAGS) -o ccl

ccl_fast: ./src/ccl_fast.cu
	$(CC) $(CFLAGS) $(SHAREDC) ./src/ccl_fast.cu $(INC) $(LDFLAGS) -o ccl_fast

clean:
	rm ccl ccl_fast
