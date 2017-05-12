#ifndef COMMON_CCL_H
#define COMMON_CCL_H

#include <cstring>
#include <cmath>
#include "../inc/helper_functions.h"    // includes cuda.h and cuda_runtime_api.h
#include "../inc/cpu_bitmap.h"
#include "EasyBMP.h"
#include <argp.h> //Part of GNU, for argument parsing
#include <stdbool.h> //For bool
#include <string.h> //For strcmp()
#include <stdlib.h> //For atoi()

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    CPUBitmap  *bitmap;
};

void copyBMPtoBitmap(BMP* input, CPUBitmap* output);
void copyBitmapToBMP(CPUBitmap* input, BMP* output);
void printMatrix(int* matrix, int width, int height);
void printMatrix(int** matrix, int width, int height);
void bitmapToBinary(CPUBitmap* input, int *output);
void binaryToBitmap(int *input, CPUBitmap* output);
void imageToBitmap(int* image, CPUBitmap* output);
void getLabelColours(int** labelColours, int maxLabels);
void markEquivalent(int** equivalenceMatrix, int a, int b);
void printArray(int* array, int size);
void updateLabelArray(int* labelArray, int** L, int maxLabel);
void resolveEquivalences(int** L, int maxLabel);
void updateRegion(int* region, int* labelArray, int width, int height);
void printLabels(int* region, int width, int height);
void label(int* region, CPUBitmap* output, int width, int height);
void anim_gpu( DataBlock *d, int ticks );
void anim_exit( DataBlock *d );

#endif
