/************************\
          783135
\************************/
// convoluteImage.cu

/*
 * This file uses code from the simpleTexture CUDA sample, copyright 
 * 1993-2015 NVIDIA Corporation. All rights reserved.
 *
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "lena_bw.pgm";

const char *sampleName = "Convolution test";

//------------------------------------------------------------------------------
// Filters
//------------------------------------------------------------------------------

// identity filter - returns the same image; used as a sanity check
float identity[1] = { 1 };

// 3x3 uniform smoothing filter - normalised
float smooth3[9] = {1.0/9.0,1.0/9.0,1.0/9.0,
                    1.0/9.0,1.0/9.0,1.0/9.0,
                    1.0/9.0,1.0/9.0,1.0/9.0};

// 5x5 uniform smoothing filter - normalised
float smooth5[25] = {1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                     1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                     1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                     1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                     1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0};
                       
// 7x7 uniform smoothing filter - normalised
float smooth7[49] = {1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,
                     1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0,1.0/49.0};
                     
 // 3x3 vertical edge detection
float vEdges[9] = {-1.0,0.0,1.0,
                  -2.0,0.0,1.0,
                  -1.0,0.0,1.0};
                   
// 3x3 horizontal edge detection
float hEdges[9] = {-1.0,-2.0,-1.0,
                   0.0, 0.0, 0.0,
                   1.0, 2.0, 1.0};
                    
// 3x3 Laplacian (edge detection)
float laplacian[9] = {-1.0,-1.0,-1.0,
                      -1.0, 8.0,-1.0,
                      -1.0,-1.0,-1.0};
                      
// 3x3 sharpening filter (based on laplacian)
float sharpen[9] = {-1.0,-1.0,-1.0,
                    -1.0, 9.0,-1.0,
                    -1.0,-1.0,-1.0};
                    
// 3x3 embossing filter (harsh)
float emboss1[9] = {2.0, 0.0, 0.0,
                    0.0,-1.0, 0.0,
                    0.0, 0.0,-1.0};
                    
// 3x3 embossing filter
float emboss2[9] = {-2.0,-1.0, 0.0,
                    -1.0, 1.0, 1.0,
                     0.0, 1.0, 2.0};
                     
// constant memory -- supports up to 32x32 filters
__constant__ float constantFilter[1024];
                      
//------------------------------------------------------------------------------
// Serial Convolution
//------------------------------------------------------------------------------
__host__ void convoluteSerial(float* input, float* output, int width, int height, float* filter, int fsize) {
    // assuming square filter with odd dimensions (i.e. side length is odd)
    int radius = fsize/2;
    int yi, xj;
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            float sum = 0;
            for(int i = -radius; i <= radius; i++) {
                for(int j = -radius; j <= radius; j++) { 
                    yi = row-i;
                    xj = col-j;
                    if(yi >= 0 && yi < height && 
                       xj >= 0 && xj < width) {
                        sum = sum + input[yi*width+xj]*filter[(i+radius)*fsize+(j+radius)];
                        //printf("%.2f*%.2f +",filter[(i+radius)*fsize+(j+radius)],input[yi*width+xj]);
                    }
                    // assume halo cells are zero (so they don't affect the sum)
                }
            }
            //printf(" = %.2f\n",sum);
            if(sum < 0) sum = 0.0;
            if(sum > 1) sum = 1.0;
            output[row*width + col] = sum;
        }
    }
}

//------------------------------------------------------------------------------
// Global Memory Convolution
//------------------------------------------------------------------------------
__global__ void convoluteGlobal(float* input, float* output, int width, int height, float* filter, int filterSize) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    int radius = filterSize/2;
    int yi, xj;
    float sum = 0;
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            yi = y-i;
            xj = x-j;
            if(yi >= 0 && yi < height && xj >= 0 && xj < width) {
                sum = sum + input[yi*width+xj]*filter[(i+radius)*filterSize+(j+radius)];
                //printf("%.2f*%.2f +",filter[(i+radius)*fsize+(j+radius)],input[yi*width+xj]);
            }
            // assume halo cells are zero (so they don't affect the sum)
        }
    }
    //printf(" = %.2f\n",sum);
    if(sum < 0) sum = 0.0;
    if(sum > 1) sum = 1.0;
    output[y*width + x] = sum;
}

//------------------------------------------------------------------------------
// Shared Memory Convolution
//------------------------------------------------------------------------------
__global__ void convoluteShared(float* input, float* output, int width, int height, float* filter, int filterSize) {
    int radius = filterSize/2;
    
    // need to load cells around the block as well and account for halo cells,
    // so shared memory array padded on all sides filterSize/2
    unsigned int sharedWidth = blockDim.x+filterSize-1;
    unsigned int sharedHeight = blockDim.y+filterSize-1;
    extern __shared__ float sdata[];
    
    // load shared pixels for block
    // first block of elements (1 per thread, offset by the filter size)
    unsigned int sharedIndex = threadIdx.y*blockDim.x + threadIdx.x;
    unsigned int sy = sharedIndex/sharedWidth;
    unsigned int sx = sharedIndex%sharedWidth;
    int Iy = blockIdx.y*blockDim.y + sy - radius;
    int Ix = blockIdx.x*blockDim.x + sx - radius;
    if(Ix >= 0 && Ix < width && Iy >= 0 && Iy < height) {
        sdata[sy*sharedWidth+sx] = input[Iy*width+Ix];
    } else {
        // treat halo cells as 0
        sdata[sy*sharedWidth+sx] = 0;
    }
    // second set of elements (up to 1 per thread)
    // -- block dimensions have to chosen such that filterSize^2 <= 2*blockDim.x*blockDim.y 
    sharedIndex = threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y;
    sy = sharedIndex/sharedWidth;
    sx = sharedIndex%sharedWidth;
    Iy = blockIdx.y*blockDim.y + sy - radius;
    Ix = blockIdx.x*blockDim.x + sx - radius;
    if (sy < sharedHeight) {
        if(Ix >= 0 && Ix < width && Iy >= 0 && Iy < height) {
            sdata[sy*sharedWidth+sx] = input[Iy*width+Ix];
        } else {
            // treat halo cells as 0
            sdata[sy*sharedWidth+sx] = 0;
        }
    }
    __syncthreads();
    
    // perform convolution
    float sum = 0;
    // pixel position in shared memory (offset by filter since array padded)
    sx = threadIdx.x+radius;
    sy = threadIdx.y+radius;
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            sum = sum + sdata[(sy+i)*sharedWidth+(sx+j)]*filter[(i+radius)*filterSize+(j+radius)];
        }
    }
    
    // clip out-of-range values
    if(sum < 0) sum = 0.0;
    if(sum > 1) sum = 1.0;
    
    // output to global memory
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    output[y*width + x] = sum;
}

//------------------------------------------------------------------------------
// Constant Global Memory Convolution
//------------------------------------------------------------------------------
__global__ void convoluteConstantGlobal(float* input, float* output, int width, int height, int filterSize) {
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    int radius = filterSize/2;
    int yi, xj;
    float sum = 0;
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            yi = y-i;
            xj = x-j;
            if(yi >= 0 && yi < height && xj >= 0 && xj < width) {
                sum = sum + input[yi*width+xj]*constantFilter[(i+radius)*filterSize+(j+radius)];
            }
            // assume halo cells are zero (so they don't affect the sum)
        }
    }
    if(sum < 0) sum = 0.0;
    if(sum > 1) sum = 1.0;
    output[y*width + x] = sum;
}

//------------------------------------------------------------------------------
// Constant Shared Memory Convolution
//------------------------------------------------------------------------------
__global__ void convoluteConstantShared(float* input, float* output, int width, int height, int filterSize) {
    int radius = filterSize/2;
    
    // need to load cells around the block as well and account for halo cells,
    // so shared memory array padded on all sides filterSize/2
    unsigned int sharedWidth = blockDim.x+filterSize-1;
    unsigned int sharedHeight = blockDim.y+filterSize-1;
    extern __shared__ float sdata[];
    
    // load shared pixels for block
    // first block of elements (1 per thread, offset by the filter size)
    unsigned int sharedIndex = threadIdx.y*blockDim.x + threadIdx.x;
    unsigned int sy = sharedIndex/sharedWidth;
    unsigned int sx = sharedIndex%sharedWidth;
    int Iy = blockIdx.y*blockDim.y + sy - radius;
    int Ix = blockIdx.x*blockDim.x + sx - radius;
    if(Ix >= 0 && Ix < width && Iy >= 0 && Iy < height) {
        sdata[sy*sharedWidth+sx] = input[Iy*width+Ix];
    } else {
        // treat halo cells as 0
        sdata[sy*sharedWidth+sx] = 0;
    }
    // second set of elements (up to 1 per thread)
    sharedIndex = threadIdx.y*blockDim.x + threadIdx.x + blockDim.x*blockDim.y;
    sy = sharedIndex/sharedWidth;
    sx = sharedIndex%sharedWidth;
    Iy = blockIdx.y*blockDim.y + sy - radius;
    Ix = blockIdx.x*blockDim.x + sx - radius;
    if (sy < sharedHeight) {
        if(Ix >= 0 && Ix < width && Iy >= 0 && Iy < height) {
            sdata[sy*sharedWidth+sx] = input[Iy*width+Ix];
        } else {
            // treat halo cells as 0
            sdata[sy*sharedWidth+sx] = 0;
        }
    }
    __syncthreads();
    
    // perform convolution
    float sum = 0;
    // pixel position in shared memory (offset by filter since array padded)
    sx = threadIdx.x+radius;
    sy = threadIdx.y+radius;
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            sum = sum + sdata[(sy+i)*sharedWidth+(sx+j)]*constantFilter[(i+radius)*filterSize+(j+radius)];
        }
    }
    
    // clip out-of-range values
    if(sum < 0) sum = 0.0;
    if(sum > 1) sum = 1.0;
    
    // output to global memory
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    output[y*width + x] = sum;
}

//------------------------------------------------------------------------------
// Texture Memory Convolution
//------------------------------------------------------------------------------
texture<float, 2, cudaReadModeElementType> texImage;

__global__ void convoluteTexture(float* output, int width, int height, int filterSize) {
    int radius = filterSize/2;
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    // perform convolution
    float sum = 0;
    int xj,yi;
    for(int i = -radius; i <= radius; i++) {
        for(int j = -radius; j <= radius; j++) {
            xj = x-j;
            yi = y-i;
            // cudaAddressModeBorder ensures halo cells 0
            sum = sum + tex2D(texImage, xj, yi)*constantFilter[(i+radius)*filterSize+(j+radius)];
        }
    }
    
    // clip out-of-range values
    if(sum < 0) sum = 0.0;
    if(sum > 1) sum = 1.0;
    
    // output to global memory
    
    output[y*width + x] = sum;
}

// test functions
bool runSerial(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);
bool runGlobal(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);
bool runShared(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);
bool runConstantGlobal(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);
bool runConstantShared(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);
bool runTexture(int argc, char** argv, float* filter, int filterSize, char* outputSuffix);

// whether to compare against a file on disk
bool reference = false;

//------------------------------------------------------------------------------
// MAIN
//------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    printf("%s starting...\n", argv[0]);
    int convolutionMethod = 4;
    // Process command-line arguments
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "input")) {
            getCmdLineArgumentString(argc,
                                     (const char **) argv,
                                     "input",
                                     (char **) &imageFilename);

            if (checkCmdLineFlag(argc, (const char **) argv, "reference")) {
                getCmdLineArgumentString(argc,
                                         (const char **) argv,
                                         "reference",
                                         (char **) &refFilename);
                reference = true;
            }
        } else if (checkCmdLineFlag(argc, (const char **) argv, "reference")) {
            printf("-reference flag should be used with -input flag");
            exit(EXIT_FAILURE);
        }
        if (checkCmdLineFlag(argc, (const char **) argv, "serial"))
            convolutionMethod = 0;
        else if (checkCmdLineFlag(argc, (const char **) argv, "global"))
            convolutionMethod = 1;
        else if (checkCmdLineFlag(argc, (const char **) argv, "shared"))
            convolutionMethod = 2;
        else if (checkCmdLineFlag(argc, (const char **) argv, "constantglobal"))
            convolutionMethod = 3;
        else if (checkCmdLineFlag(argc, (const char **) argv, "constantshared"))
            convolutionMethod = 4;
        else if (checkCmdLineFlag(argc, (const char **) argv, "texture"))
            convolutionMethod = 5;
    }

    int option, ignore;
    do {
        printf("\nOptions:\n");
        printf("0: None\n");
        printf("1: 3x3 Uniform Smooth\n");
        printf("2: 5x5 Uniform Smooth\n");
        printf("3: 7x7 Uniform Smooth\n");
        printf("4: Vertical Edges\n");
        printf("5: Horizontal Edges\n");
        printf("6: Laplacian Edges\n");
        printf("7: Sharpen\n");
        printf("8: Emboss v1\n");
        printf("9: Emboss v2\n");
        printf("Q: Quit\n");
        printf("Select an option: ");
        do{
            option = getchar();
        } while(isspace(option));
        //fflush(stdin);
        // flush stdin
        while((ignore = getchar()) != '\n' && ignore != EOF);
        
        float* filter = NULL;
        int filterSize;
        char suffix[10];
        switch(option) {
            case '0':
                filter = identity;
                filterSize = 1;
                strncpy(suffix,"_identity",10);
                break;
            case '1':
                filter = smooth3;
                filterSize = 3;
                strncpy(suffix,"_smooth3",10);
                break;
            case '2':
                filter = smooth5;
                filterSize = 5;
                strncpy(suffix,"_smooth5",10);
                break;
            case '3':
                filter = smooth7;
                filterSize = 7;
                strncpy(suffix,"_smooth7",10);
                break;
            case '4':
                filter = vEdges;
                filterSize = 3;
                strncpy(suffix,"_vedges",10);
                break;
            case '5':
                filter = hEdges;
                filterSize = 3;
                strncpy(suffix,"_hedges",10);
                break;
            case '6':
                filter = laplacian;
                filterSize = 3;
                strncpy(suffix,"_edges",10);
                break;
            case '7':
                filter = sharpen;
                filterSize = 3;
                strncpy(suffix,"_sharpen",10);
                break;
            case '8':
                filter = emboss1;
                filterSize = 3;
                strncpy(suffix,"_embossv1",10);
                break;
            case '9':
                filter = emboss2;
                filterSize = 3;
                strncpy(suffix,"_embossv2",10);
                break;
            case 'q':
            case 'Q':
                break;
            default:
                printf("Invalid option!\n");
        };
        
        // perform covolution
        if(filter != NULL) {
            if(convolutionMethod == 0) {
                runSerial(argc,argv,filter,filterSize,suffix);
            } else if(convolutionMethod == 1) {
                runGlobal(argc,argv,filter,filterSize,suffix);
                cudaDeviceReset();
            } else if(convolutionMethod == 2) {
                runShared(argc,argv,filter,filterSize,suffix);
                cudaDeviceReset();
            } else if(convolutionMethod == 3) {
                runConstantGlobal(argc,argv,filter,filterSize,suffix);
                cudaDeviceReset();
            } else if(convolutionMethod == 4) {
                runConstantShared(argc,argv,filter,filterSize,suffix);
                cudaDeviceReset();
            } else if(convolutionMethod == 5) {
                runTexture(argc,argv,filter,filterSize,suffix);
                cudaDeviceReset();
            }
            printf("Press ENTER to continue...");
            // flush stdin
            while((ignore = getchar()) != '\n' && ignore != EOF);
        }
    } while(option != 'Q' && option != 'q');

    return 0;
}

//------------------------------------------------------------------------------
// Serial Test
//------------------------------------------------------------------------------
bool runSerial(int argc, char** argv, float* filter, int filterSize, char* outputSuffix) {
    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);  

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);

    double startd = (double)clock() * 1000.0f / (double)CLOCKS_PER_SEC;
    // Convolute
    convoluteSerial(hData,hOutputData,width,height,filter,filterSize);
    double endd = (double)clock() * 1000.0f / (double)CLOCKS_PER_SEC;
    printf("Processing time: %f (ms)\n",endd-startd);
    printf("%.2f Mpixels/sec\n",width*height/((endd-startd)/1000.0f)/1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / ((endd-startd) / 1000.0f)) / 1e9);
    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
    }
    return testResult;
}

//------------------------------------------------------------------------------
// Global Memory GPU Test
//------------------------------------------------------------------------------
bool runGlobal(int argc, char** argv, float* filter, int filterSize, char* outputSuffix)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float* dData = NULL;
    float* dOutput = NULL;
    float* dFilter = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &dOutput, size));
    checkCudaErrors(cudaMalloc((void **) &dFilter, filterSize*filterSize*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dFilter,filter,filterSize*filterSize*sizeof(float),cudaMemcpyHostToDevice));    

    // 8x8 blocks
    dim3 dimBlock(32, 32, 1);
    // ceiling calculations to divide image into blocks if size not exactly divisible
    dim3 dimGrid(((width-1) / dimBlock.x)+1, ((height-1) / dimBlock.y)+1, 1);

    // Warmup
    convoluteGlobal<<<dimGrid, dimBlock, 0>>>(dData, dOutput, width, height, dFilter, filterSize);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convoluteGlobal<<<dimGrid, dimBlock, 0>>>(dData, dOutput, width, height, dFilter, filterSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e9);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dOutput,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);
    
    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
        free(refPath);
    }
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(dOutput));
    checkCudaErrors(cudaFree(dFilter));
    free(imagePath);
    return testResult;
}

//------------------------------------------------------------------------------
// Constant Global Memory GPU Test
//------------------------------------------------------------------------------
bool runConstantGlobal(int argc, char** argv, float* filter, int filterSize, char* outputSuffix)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float* dData = NULL;
    float* dOutput = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &dOutput, size));
    checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));
    // copy filter to constant memory
    cudaMemcpyToSymbol(constantFilter, filter, filterSize*filterSize*sizeof(float));   

    // 8x8 blocks
    dim3 dimBlock(32, 32, 1);
    // ceiling calculations to divide image into blocks if size not exactly divisible
    dim3 dimGrid(((width-1) / dimBlock.x)+1, ((height-1) / dimBlock.y)+1, 1);

    // Warmup
    convoluteConstantGlobal<<<dimGrid, dimBlock, 0>>>(dData, dOutput, width, height, filterSize);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convoluteConstantGlobal<<<dimGrid, dimBlock, 0>>>(dData, dOutput, width, height, filterSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e9);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dOutput,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);
    
    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
        free(refPath);
    }
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(dOutput));
    free(imagePath);
    return testResult;
}

//------------------------------------------------------------------------------
// Shared Memory GPU Test
//------------------------------------------------------------------------------
bool runShared(int argc, char** argv, float* filter, int filterSize, char* outputSuffix)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float* dData = NULL;
    float* dOutput = NULL;
    float* dFilter = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &dOutput, size));
    checkCudaErrors(cudaMalloc((void **) &dFilter, filterSize*filterSize*sizeof(float)));
    checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dFilter,filter,filterSize*filterSize*sizeof(float),cudaMemcpyHostToDevice));    

    // calculate block size so each thread loads at most 2 pixels to shared memory
    unsigned int bw = 32, bh = 32;
    while(2*bw*bh < (bw+filterSize-1)*(bh+filterSize-1)) {
        bw = bh = 2*bw;
    }
    dim3 dimBlock(bw, bh, 1);
    // ceiling calculations to divide image into blocks if size not exactly divisible
    dim3 dimGrid(((width-1) / dimBlock.x)+1, ((height-1) / dimBlock.y)+1, 1);
    
    // shared memory size 
    // need to load cells around the block as well and account for halo cells,
    // so shared memory array padded on all sides filterSize/2
    unsigned int sharedWidth = dimBlock.x+filterSize-1;
    unsigned int sharedHeight = dimBlock.y+filterSize-1;
    unsigned int sharedSize = sharedWidth*sharedHeight*sizeof(float);

    // Warmup
    convoluteShared<<<dimGrid, dimBlock, sharedSize>>>(dData, dOutput, width, height, dFilter, filterSize);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convoluteShared<<<dimGrid, dimBlock, sharedSize>>>(dData, dOutput, width, height, dFilter, filterSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e9);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dOutput,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);
    
    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
        free(refPath);
    }
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(dOutput));
    checkCudaErrors(cudaFree(dFilter));
    free(imagePath);
    return testResult;
}

//------------------------------------------------------------------------------
// Constant Shared Memory GPU Test
//------------------------------------------------------------------------------
bool runConstantShared(int argc, char** argv, float* filter, int filterSize, char* outputSuffix)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float* dData = NULL;
    float* dOutput = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    checkCudaErrors(cudaMalloc((void **) &dOutput, size));
    checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));
    // copy filter to constant memory
    cudaMemcpyToSymbol(constantFilter, filter, filterSize*filterSize*sizeof(float));

    // calculate block size so each thread loads at most 2 pixels to shared memory
    unsigned int bw = 32, bh = 32;
    while(2*bw*bh < (bw+filterSize-1)*(bh+filterSize-1)) {
        bw = bh = 2*bw;
    }
    dim3 dimBlock(bw, bh, 1);
    // ceiling calculations to divide image into blocks if size not exactly divisible
    dim3 dimGrid(((width-1) / dimBlock.x)+1, ((height-1) / dimBlock.y)+1, 1);
    
    // shared memory size 
    // need to load cells around the block as well and account for halo cells,
    // so shared memory array padded on all sides filterSize/2
    unsigned int sharedWidth = dimBlock.x+filterSize-1;
    unsigned int sharedHeight = dimBlock.y+filterSize-1;
    unsigned int sharedSize = sharedWidth*sharedHeight*sizeof(float);

    // Warmup
    convoluteConstantShared<<<dimGrid, dimBlock, sharedSize>>>(dData, dOutput, width, height, filterSize);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convoluteConstantShared<<<dimGrid, dimBlock, sharedSize>>>(dData, dOutput, width, height, filterSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e9);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dOutput,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);
    
    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
        free(refPath);
    }

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFree(dOutput));
    free(imagePath);
    return testResult;
}


//------------------------------------------------------------------------------
// Texture Memory GPU Test
//------------------------------------------------------------------------------
bool runTexture(int argc, char** argv, float* filter, int filterSize, char* outputSuffix) {
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    // Allocate device memory for result
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, size));
    // copy filter to constant memory
    cudaMemcpyToSymbol(constantFilter, filter, filterSize*filterSize*sizeof(float));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      hData,
                                      size,
                                      cudaMemcpyHostToDevice));

    // Set texture parameters
    // cudaAddressModeBorder so halo cells are 0
    texImage.addressMode[0] = cudaAddressModeBorder;
    texImage.addressMode[1] = cudaAddressModeBorder;
    texImage.filterMode = cudaFilterModePoint; // don't want any interpolation, only exact pixel values
    texImage.normalized = false; // do NOT use normalised texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(texImage, cuArray, channelDesc));

    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // Warmup
    convoluteTexture<<<dimGrid, dimBlock, 0>>>(dData, width, height, filterSize);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
    convoluteTexture<<<dimGrid, dimBlock, 0>>>(dData, width, height, filterSize);

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    printf("%.2f GFlops\n",
           (filterSize*filterSize*width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e9);
    sdkDeleteTimer(&timer);

    // Allocate mem for the result on host side
    float *hOutputData = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(hOutputData,
                               dData,
                               size,
                               cudaMemcpyDeviceToHost));

    // Write result to file
    char outputFilename[1024];
    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, outputSuffix);
    strcat(outputFilename, ".pgm");
    sdkSavePGM(outputFilename, hOutputData, width, height);
    printf("Wrote '%s'\n", outputFilename);

    bool testResult = true;
    if(reference) {
        // We need to reload the data from disk,
        // because it is inverted upon output
        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
        
        //Load reference image from image (output)
        float *hDataRef = (float *) malloc(size);
        char *refPath = sdkFindFilePath(refFilename, argv[0]);

        if (refPath == NULL)
        {
            printf("Unable to find reference image file: %s\n", refFilename);
            exit(EXIT_FAILURE);
        }

        sdkLoadPGM(refPath, &hDataRef, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", outputFilename);
        printf("\treference: <%s>\n", refPath);

        testResult = compareData(hOutputData,
                             hDataRef,
                             width*height,
                             MAX_EPSILON_ERROR,
                             0.15f);
        printf("Test completed, result %s\n", testResult ? "OK" : "ERROR!");
        free(refPath);
    }
    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    free(imagePath);
    return testResult;
}
