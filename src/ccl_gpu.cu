
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>
#include <cstring> 
#include <cmath>
#include "cuda.h"
#include "book.h" //HANDLE_ERROR
#include "cpu_bitmap.h" 

#include "EasyBMP.h"

int regionWidth = 8;
int regionHeight = 8;
int total_index;

//Texture binding variable
surface<void, cudaSurfaceType2D> surf_ref; 

#define cudaErrorCheck(t) { \
 t; \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
}

// globals needed by the update routine
struct DataBlock {
    unsigned char   *output_bitmap;
    float           *dev_inSrc;
    float           *dev_outSrc;
    float           *dev_constSrc;
    //CPUAnimBitmap  *bitmap;
    CPUBitmap  *bitmap;

    cudaEvent_t     start, stop;
    float           totalTime;
    float           frames;
};

void anim_gpu( DataBlock *d, int ticks ) {
    /*HANDLE_ERROR( cudaEventRecord( d->start, 0 ) );
    dim3    blocks(DIM/16,DIM/16);
    dim3    threads(16,16);
    CPUAnimBitmap  *bitmap = d->bitmap;

    // since tex is global and bound, we have to use a flag to
    // select which is in/out per iteration
    volatile bool dstOut = true;
    for (int i=0; i<90; i++) {
        float   *in, *out;
        if (dstOut) {
            in  = d->dev_inSrc;
            out = d->dev_outSrc;
        } else {
            out = d->dev_inSrc;
            in  = d->dev_outSrc;
        }
        copy_const_kernel<<<blocks,threads>>>( in );
        blend_kernel<<<blocks,threads>>>( out, dstOut );
        dstOut = !dstOut;
    }
    float_to_color<<<blocks,threads>>>( d->output_bitmap,
                                        d->dev_inSrc );

    HANDLE_ERROR( cudaMemcpy( bitmap->get_ptr(),
                              d->output_bitmap,
                              bitmap->image_size(),
                              cudaMemcpyDeviceToHost ) );

    HANDLE_ERROR( cudaEventRecord( d->stop, 0 ) );
    HANDLE_ERROR( cudaEventSynchronize( d->stop ) );
    float   elapsedTime;
    HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        d->start, d->stop ) );
    d->totalTime += elapsedTime;
    ++d->frames;
    printf( "Average Time per frame:  %3.1f ms\n",
            d->totalTime/d->frames  );*/
}

// clean up memory allocated on the GPU
void anim_exit( DataBlock *d ) {
    /*
    cudaUnbindTexture( texIn );
    cudaUnbindTexture( texOut );
    cudaUnbindTexture( texConstSrc );
    HANDLE_ERROR( cudaFree( d->dev_inSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_outSrc ) );
    HANDLE_ERROR( cudaFree( d->dev_constSrc ) );

    HANDLE_ERROR( cudaEventDestroy( d->start ) );
    HANDLE_ERROR( cudaEventDestroy( d->stop ) );
    */
}

void printMatrix(int* matrix, int width, int height) {
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int val = matrix[y*width+x];
			if(val < 10) {
				printf(" %d ",val);
			} else {
				printf("%d ",val);
			}
		}
		printf("\n");
	}
}

void copyBMPtoBitmap(BMP* input, CPUBitmap* output) {
	unsigned char *rgbaPixels = output->get_ptr();
	int width = input->TellWidth();
	int height = input->TellHeight();
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			rgbaPixels[y*4*width+4*x]   = (*input)(x,y)->Red;
			rgbaPixels[y*4*width+4*x+1] = (*input)(x,y)->Green;
			rgbaPixels[y*4*width+4*x+2] = (*input)(x,y)->Blue;
			rgbaPixels[y*4*width+4*x+3] = (*input)(x,y)->Alpha;
		}
	}
}

void copyBitmapToBMP(CPUBitmap* input, BMP* output) {
	unsigned char *rgbaPixels = input->get_ptr();
	int width = input->x;
	int height = input->y;
	output->SetSize(width,height);
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			(*output)(x,y)->Red   = rgbaPixels[y*4*width+4*x];
		    (*output)(x,y)->Green = rgbaPixels[y*4*width+4*x+1];
			(*output)(x,y)->Blue  = rgbaPixels[y*4*width+4*x+2];
			(*output)(x,y)->Alpha = rgbaPixels[y*4*width+4*x+3];
		}
	}
}

void bitmapToBinary(CPUBitmap* input, int *output) {
	unsigned char *rgbaPixels = input->get_ptr();
	int width = input->x;
	int height = input->y;
	// output should be of size width*height
	// assuming 4 byte stride for RGBA values
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			unsigned char r = rgbaPixels[y*4*width+4*x];
			unsigned char g = rgbaPixels[y*4*width+4*x+1];
			unsigned char b = rgbaPixels[y*4*width+4*x+2];
			// Thresholding according to: (r+g+b)/3 > 128
			output[y*width+x] = ((r+g+b) > 348); // 1 -> white, 0 -> black
		}
	}
}

void binaryToBitmap(int *input, CPUBitmap* output) {
	unsigned char *rgbaPixels = output->get_ptr();
	int width = output->x;
	int height = output->y;
	// assuming 4 byte stride for RGBA values
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			// [0,1] -> [0,255]
			rgbaPixels[y*4*width+4*x]   = input[y*width+x]*255;
			rgbaPixels[y*4*width+4*x+1] = input[y*width+x]*255;
			rgbaPixels[y*4*width+4*x+2] = input[y*width+x]*255;
			rgbaPixels[y*4*width+4*x+3] = 255;
		}
	}
}

void colourise(int* input, CPUBitmap* output, int width, int height) {
	unsigned char *rgbaPixels = output->get_ptr();
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int label = input[y*width+x];
			if(label == 0) {
				rgbaPixels[y*4*width+4*x]   = 0;
				rgbaPixels[y*4*width+4*x+1] = 0;
				rgbaPixels[y*4*width+4*x+2] = 0;
				rgbaPixels[y*4*width+4*x+3] = 255;
				continue;
			}
			rgbaPixels[y*4*width+4*x]   = input[y*width+x] * 131 % 255;
			rgbaPixels[y*4*width+4*x+1] = input[y*width+x] * 241 % 255;
			rgbaPixels[y*4*width+4*x+2] = input[y*width+x] * 251 % 255;
			rgbaPixels[y*4*width+4*x+3] = 255;
		}
	}
}

__global__ void gpu_label(int width, int height) {
    // STEP 1 - Initial Labelling

	//From https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
	//int i = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = width*y+x+1; // +1 to avoid 0 labels

	int temp;
    if ((x<width) && (y<height)) {
        surf2Dread(&temp, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(temp != 0) {
	        surf2Dwrite(idx, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        }
        //printf("x = %d, y = %d, i = %d\n",x,y,idx);
	}
}

__device__ int getMinNeighbourScan(int x, int y, int label) {
    // boundary mode zero causes out of range reads to return 0 (convenient)
    int minLabel = label, curr;
    // south-west
    surf2Dread(&curr, surf_ref, (x-1)*sizeof(int), y+1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // west
    surf2Dread(&curr, surf_ref, (x-1)*sizeof(int), y, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-west
    surf2Dread(&curr, surf_ref, (x-1)*sizeof(int), y-1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // north
    surf2Dread(&curr, surf_ref, x*sizeof(int), y-1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-east
    surf2Dread(&curr, surf_ref, (x+1)*sizeof(int), y-1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    return minLabel;
}

__global__ void gpu_scan(int width, int height) {
    // STEP 2 - Scanning
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int label;
    if ((x<width) && (y<height)) {
	    surf2Dread(&label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(label != 0) {
            int minLabelScanned = getMinNeighbourScan(x,y,label);
	        surf2Dwrite(minLabelScanned, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        }
	}
}

__global__ void gpu_analysis(int width, int height) {
    // STEP 3 - Analysis
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
	    surf2Dread(&label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(label != 0) {
            // propagate labels
            // "recursively" get the final label
            // - if first referred pixel index refers to another label
            // - stop when the label refers to itself
            int idx = -1;
            int lx,ly;
            while(label != (idx+1)) {
                idx = label-1; // -1 since labels start from 1 and we want 1D pixel index
                lx = idx%width;
                ly = idx/width;
                surf2Dread(&label, surf_ref, lx*sizeof(int), ly, cudaBoundaryModeZero);
            }
	        surf2Dwrite(label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        }
	}
}

__device__ int getMinNeighbourLink(int x, int y, int label) {
    // boundary mode zero causes out of range reads to return 0 (convenient)
    int minLabel = label, curr;
    // south-west
    surf2Dread(&curr, surf_ref, (x-1)*sizeof(int), y+1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // west
    surf2Dread(&curr, surf_ref, (x-1)*sizeof(int), y, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // east
    surf2Dread(&curr, surf_ref, (x+1)*sizeof(int), y, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-east
    surf2Dread(&curr, surf_ref, (x+1)*sizeof(int), y-1, cudaBoundaryModeZero);
    if(curr > 0) minLabel = min(minLabel,curr);
    return minLabel;
}

__global__ void gpu_link(int width, int height) {
    // STEP 4 - Link
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
	    surf2Dread(&label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(label != 0) {
            // scan neighbours
            int minLabel = getMinNeighbourLink(x,y,label);
            // update pixel of REFERENCE label (not current pixel)
            // this is so that all other pixels can simply reference that pixel
            // in the next step
            if(minLabel < label) {
                int refIdx = label-1; // -1 since labels start from 1 and we want 1D pixel index
                int refx = refIdx%width;
                int refy = refIdx/width;
	            surf2Dwrite(minLabel, surf_ref, refx*sizeof(int), refy, cudaBoundaryModeZero);
            }
        }
	}
}

__global__ void gpu_relabel(int width, int height) {
    // STEP 5 - Re-label
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
	    surf2Dread(&label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(label != 0) {
            // resolve label equivalences (after previous step)
            int refIdx = label-1; // -1 since labels start from 1 and we want 1D pixel index
            int refx = refIdx%width;
            int refy = refIdx/width;
            int refLabel;
            surf2Dread(&refLabel, surf_ref, refx*sizeof(int), refy, cudaBoundaryModeZero);
            surf2Dwrite(refLabel, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        }
	}
}

__device__ bool done;
__global__ void gpu_rescan(int width, int height) {
    // STEP 5 - Re-Scan
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
	    surf2Dread(&label, surf_ref, x*sizeof(int), y, cudaBoundaryModeZero);
        if(label != 0) {
            // check if all regions are connected
            int minNeighbour = getMinNeighbourScan(x,y,label);
            if(minNeighbour != label) {
                done = false;
            }
        }
	}
}

void gpu_label(int* image, CPUBitmap* output, int width, int height) {
    cudaArray* gpuImage;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
    cudaErrorCheck(cudaMallocArray(&gpuImage, &channelDesc, width, height, cudaArraySurfaceLoadStore));
    cudaErrorCheck(cudaMemcpyToArray(gpuImage, 0, 0, image, width*height*sizeof(int), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaBindSurfaceToArray(surf_ref, gpuImage));
    

    dim3 block_dim(regionWidth, regionHeight);
    int gridWidth = width/block_dim.x;
    int gridHeight = height/block_dim.y;
    if (width%block_dim.x != 0) gridWidth++;
    if (height%block_dim.y != 0) gridHeight++;
    int result = false;
    dim3 grid_dim(gridWidth, gridHeight);
    //printf("Initial...\n");
    //printMatrix(image,width,height);
    gpu_label<<<grid_dim, block_dim>>>(width, height);
    //cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
    //printf("AFTER LABELLING...\n");
    //printMatrix(image,width,height);
    gpu_scan<<<grid_dim, block_dim>>>(width, height);
    //cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
    //printf("AFTER SCAN...\n");
    //printMatrix(image,width,height);
    gpu_analysis<<<grid_dim, block_dim>>>(width, height);
    //cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
    while(result == false) {
        //printf("AFTER ANALYSIS...\n");
        //printMatrix(image,width,height);
        gpu_link<<<grid_dim, block_dim>>>(width, height);
        //cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
        //printf("AFTER LINK...\n");
        //printMatrix(image,width,height);
        gpu_relabel<<<grid_dim, block_dim>>>(width, height);
        //cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
        //printf("AFTER RE-LABELLING...\n");
        //printMatrix(image,width,height);
        result = true;
        cudaErrorCheck(cudaMemcpyToSymbol(done, &result, sizeof(bool)));
        gpu_rescan<<<grid_dim, block_dim>>>(width, height);
        cudaErrorCheck(cudaMemcpyFromSymbol(&result, done, sizeof(bool)));
    }
    cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
    // apparently you don't have to unbind surfaces.
    cudaErrorCheck(cudaFreeArray(gpuImage));
    
    //Colourise
    colourise(image,output,width,height);
}

int main(int argc, char **argv) {
    printf("%s Starting...\n\n", argv[0]);

    //initialize CUDA
    findCudaDevice(argc, (const char **)argv);

    //source and results image filenames
    char SampleImageFname[] = "3pixeldeath.bmp";
    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

    if (pSampleImageFpath == NULL) {
        printf("%s could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    BMP input;
    BMP output;

	printf("===============================================\n");
    printf("Loading image: %s...\n", pSampleImageFpath);
    bool result = input.ReadFromFile(pSampleImageFpath);
	if (result == false) {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
        return 1;
    }
    printf("===============================================\n");

	int width = input.TellWidth();
	int height = input.TellHeight();
	output.SetSize(width,height);
	output.SetBitDepth(32); // RGBA
    DataBlock   data;
    CPUBitmap bitmap( width, height, &data );
    data.bitmap = &bitmap;
    data.totalTime = 0;
    data.frames = 0;
    //HANDLE_ERROR( cudaEventCreate( &data.start ) );
    //HANDLE_ERROR( cudaEventCreate( &data.stop ) );
    copyBMPtoBitmap(&input,&bitmap);
    int* binaryImage = new int[width*height];
    bitmapToBinary(&bitmap,binaryImage);

    printf("LABELLING...\n");
    gpu_label(binaryImage,&bitmap,width,height);
    printf("FINISHED...\n");

    copyBitmapToBMP(&bitmap,&output);
    //binaryToBitmap(binaryImage,&bitmap);
    //copyBitmapToBMP(&bitmap,&output);
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
    //DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
    output.WriteToFile("out.bmp");
    bitmap.display_and_exit((void (*)(void*))anim_exit);
}

