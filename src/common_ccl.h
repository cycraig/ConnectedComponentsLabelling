#ifndef COMMON_CCL_H
#define COMMON_CCL_H

#include <cstring>
#include <cmath>
#include "../inc/helper_functions.h"    // includes cuda.h and cuda_runtime_api.h
#include "../inc/cpu_bitmap.h"
#include "EasyBMP.h"
#include <argp.h> //Part of GNU, for argument parsing
#include <stdbool.h> //For bool
#include <string.h> //For strcmp(), strcat()
#include <stdlib.h> //For atoi()
#include <time.h> // For seeding srand()
#include <stdlib.h> //For srand(), rand()

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    CPUBitmap  *bitmap;
};

//For argument handling
typedef enum {NORMAL_MODE, RANDOM_MODE} image_mode_t;
struct arguments {
    image_mode_t mode;
		char* filename;
		bool bench;
		bool visualise;
		int width;
    int region_width;
};
static error_t parse_opt(int key, char *arg, struct argp_state *state);
bool get_args(int argc, char** argv, struct arguments* parsed_args);

bool start(int argc, char** argv,
    int& width, int& height,
    BMP& input,
    struct arguments& parsed_args);
void finish(int& width, int& height,
    BMP& ouput,
    CPUBitmap * bitmap,
    int* binaryImage,
    arguments& parsed_args);
void colourise(int* input, CPUBitmap* output, int width, int height);
void makeRandomBMP(BMP* output, int width, int height);
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
