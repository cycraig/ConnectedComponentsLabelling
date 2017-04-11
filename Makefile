CFLAGS=-O0 -g -Wno-deprecated-gpu-targets
INC=-I./inc
LDFLAGS= -lglut -lGL -lGLU
CU=./src/EasyBMP.cpp ./src/ccl.cu
EXE=ccl
CC=nvcc

all: $(EXE)

$(EXE): $(CU)
	$(CC) $(CFLAGS) $(CU) $(INC) $(LDFLAGS) -o $(EXE) 
