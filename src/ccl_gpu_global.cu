#include <cuda_runtime.h>
#include <cuda.h>
#include "../inc/helper_cuda.h"
#include "common_ccl.h"

#define cudaErrorCheck(t) { \
 t; \
 cudaError_t e=cudaGetLastError(); \
 if(e!=cudaSuccess) { \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0); \
 } \
}

int regionWidth = 32;
int regionHeight = 32;
int total_index;

__global__ void gpu_label(int width, int height, int* globalImage) {
    // STEP 1 - Initial Labelling

	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = width*y+x+1; // +1 to avoid 0 labels

	int temp;
    if ((x<width) && (y<height)) {
	temp = globalImage[x+y*width];
        if(temp != 0) {
			globalImage[x+y*width] = idx;
        }
        //printf("x = %d, y = %d, i = %d\n",x,y,idx);
	}
}

__device__ int getMinNeighbourScan(int x, int y, int width, int height, int label, int* globalImage) {
    int minLabel = label, curr = -1;
    // south-west
	if(x > 0 && y < (height-1))
    	curr = globalImage[x-1+(y+1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // west
	if(x > 0)
    	curr = globalImage[x-1+(y)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-west
	if(x > 0 && y > 0)
    	curr = globalImage[x-1+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // north
	if(y > 0)
    	curr = globalImage[x+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-east
	if(x < (width-1) && y > 0)
    	curr = globalImage[x+1+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    return minLabel;
}

__global__ void gpu_scan(int width, int height, int* globalImage) {
    // STEP 2 - Scanning
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int label;
    if ((x<width) && (y<height)) {
	    label = globalImage[x+y*width];
        if(label != 0) {
            int minLabelScanned = getMinNeighbourScan(x,y,width,height,label,globalImage);
	        globalImage[x+y*width] = minLabelScanned;
        }
	}
}

__global__ void gpu_analysis(int width, int height, int* globalImage) {
    // STEP 3 - Analysis
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    //int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
		label = globalImage[x+y*width];
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
				label = globalImage[lx+ly*width];
            }
			globalImage[x+y*width] = label;
        }
	}
}

__device__ int getMinNeighbourLink(int x, int y, int width, int height, int label, int* globalImage) {
    int minLabel = label;
	int curr = -1;

	// CHANGED FROM PAPER
	// Need to check south-east, north, and north-west as well for the algorithm to work

	// south-west
	if(x > 0 && y < (height-1))
    	curr = globalImage[x-1+(y+1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
	// south-east
	if(x < (width-1) && y < (height-1))
    	curr = globalImage[x+1+(y+1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // west
	if(x > 0)
    	curr = globalImage[x-1+(y)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // east
	if(x < (width-1))
    	curr = globalImage[x+1+(y)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    // north-east
	if(x < (width-1) && y > 0)
    	curr = globalImage[x+1+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
	// north
	if(y > 0)
		curr = globalImage[x+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
	// north-west
	if(x > 0 && y > 0)
		curr = globalImage[x-1+(y-1)*width];
    if(curr > 0) minLabel = min(minLabel,curr);
    return minLabel;
}

__global__ void gpu_link(int width, int height, int* globalImage) {
    // STEP 4 - Link
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    //int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
		label = globalImage[x+y*width];
        if(label != 0) {
            // scan neighbours
            int minLabel = getMinNeighbourLink(x,y,width, height, label, globalImage);
            // update pixel of REFERENCE label (not current pixel)
            // this is so that all other pixels can simply reference that pixel
            // in the next step
            if(minLabel < label) {
                int refIdx = label-1; // -1 since labels start from 1 and we want 1D pixel index
                int refx = refIdx%width;
                int refy = refIdx/width;
				// reduces contention - makes it faster than surface
				atomicMin(&globalImage[refx+refy*width],minLabel);
            }
        }
	}
}

__global__ void gpu_relabel(int width, int height, int* globalImage) {
    // STEP 5 - Re-label
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    //int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
		label = globalImage[x+y*width];
        if(label != 0) {
            // resolve label equivalences (after previous step)
            int refIdx = label-1; // -1 since labels start from 1 and we want 1D pixel index
            int refx = refIdx%width;
            int refy = refIdx/width;
            int refLabel;
			refLabel = globalImage[refx+refy*width];
			globalImage[x+y*width] = refLabel;
        }
	}
}

__device__ bool done;
__global__ void gpu_rescan(int width, int height, int* globalImage) {
    // STEP 5 - Re-Scan
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
    //int i = y*width + x;

	int label;
    if ((x<width) && (y<height)) {
	    label = globalImage[x+y*width];
        if(label != 0) {
            // check if all regions are connected
            int minNeighbour = getMinNeighbourScan(x,y,width,height,label,globalImage);
            if(minNeighbour != label) {
                done = false;
            }
        }
	}
}

void gpu_label(int* image, CPUBitmap* output, int width, int height, float* gpuTime) {
	int* globalImage;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindSigned);
	cudaErrorCheck(cudaMalloc((void**)&globalImage,width*height*sizeof(int)));
	cudaErrorCheck(cudaMemcpy(globalImage,image,width*height*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block_dim(regionWidth, regionHeight);
    int gridWidth = width/block_dim.x;
    int gridHeight = height/block_dim.y;
    if (width%block_dim.x != 0) gridWidth++;
    if (height%block_dim.y != 0) gridHeight++;
    bool result = false;
    dim3 grid_dim(gridWidth, gridHeight);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gpu_label<<<grid_dim, block_dim>>>(width, height, globalImage);
	cudaDeviceSynchronize();
    gpu_scan<<<grid_dim, block_dim>>>(width, height, globalImage);
	cudaDeviceSynchronize();
    gpu_analysis<<<grid_dim, block_dim>>>(width, height, globalImage);
	cudaDeviceSynchronize();
    while(result == false) {
        gpu_link<<<grid_dim, block_dim>>>(width, height, globalImage);
		cudaDeviceSynchronize();
        gpu_relabel<<<grid_dim, block_dim>>>(width, height, globalImage);
		cudaDeviceSynchronize();
        result = true;
        cudaErrorCheck(cudaMemcpyToSymbol(done, &result, sizeof(bool)));
        gpu_rescan<<<grid_dim, block_dim>>>(width, height, globalImage);
		cudaDeviceSynchronize();
        cudaErrorCheck(cudaMemcpyFromSymbol(&result, done, sizeof(bool)));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    *gpuTime = 0;
    cudaEventElapsedTime(gpuTime, start, stop);
	cudaErrorCheck(cudaMemcpy(image, globalImage,width*height*sizeof(int), cudaMemcpyDeviceToHost));
	cudaErrorCheck(cudaFree(globalImage));
}

int main(int argc, char **argv) {
  int width, height;
	int* dims = new int[2];
	int* binaryImage;
	CPUBitmap *bitmap;
	DataBlock data;
	BMP output;
  BMP input;
  struct arguments parsed_args;

  if (!start(argc, argv,
      width, height,
      input,
      parsed_args)) exit(EXIT_FAILURE);

      regionWidth = parsed_args.region_width;
      regionHeight = parsed_args.region_width;

  bitmap = new CPUBitmap( width, height, &data );
  data.bitmap = bitmap;
  copyBMPtoBitmap(&input,bitmap);
  binaryImage = new int[(width)*(height)];
  bitmapToBinary(bitmap,binaryImage);
  output.SetSize(width,height);
  output.SetBitDepth(32); // RGBA

    fprintf(stderr,"LABELLING...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

   // double start_time = omp_get_wtime();
    float gpuTime = 0;
    gpu_label(binaryImage,bitmap,width,height,&gpuTime);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    fprintf(stderr,"FINISHED...\n");
    //printf("Time elapsed: %f ms\n",(end_time-start_time)*1000.0);
    if (!parsed_args.bench) {
      printf("Time elapsed (gpu): %.6f ms\n",gpuTime);
      printf("Time elapsed (total): %.6f ms\n",milliseconds);
    }
    else {
      printf("%s,%d,%d,%f,%f\n",parsed_args.mode==NORMAL_MODE?"normal":"random",
       width*height,regionWidth*regionHeight
       gpuTime,milliseconds);
    }

    finish(width, height,
            output,
            bitmap,
            binaryImage,
            parsed_args,
            "ccl_gpu_global");
    delete[] binaryImage;
}
