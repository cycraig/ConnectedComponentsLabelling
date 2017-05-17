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

int regionWidth = 8;
int regionHeight = 8;
int total_index;

//Texture binding variable
surface<void, cudaSurfaceType2D> surf_ref;

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
			/*rgbaPixels[y*4*width+4*x]   = (input[y*width+x] * 131) % 255;
			rgbaPixels[y*4*width+4*x+1] = (input[y*width+x] * 241) % 255;
			rgbaPixels[y*4*width+4*x+2] = (input[y*width+x] * 251) % 255;*/
			rgbaPixels[y*4*width+4*x]   = (input[y*width+x] * 131) % 177 + (input[y*width+x] * 131) % 78+1;
			rgbaPixels[y*4*width+4*x+1] = (input[y*width+x] * 241) % 56 + (input[y*width+x] * 241) % 199+1;
			rgbaPixels[y*4*width+4*x+2] = (input[y*width+x] * 251) % 237  + (input[y*width+x] * 241) % 18+1;
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
    //int i = y*width + x;

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
    //int i = y*width + x;

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
    //int i = y*width + x;

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
    //int i = y*width + x;

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

void gpu_label(int* image, CPUBitmap* output, int width, int height, float* gpuTime) {
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
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    *gpuTime = 0;
    cudaEventElapsedTime(gpuTime, start, stop);
    //printf("Time elapsed (gpu):   %.6f ms\n",milliseconds);

    cudaErrorCheck(cudaMemcpyFromArray(image, gpuImage, 0, 0,width*height*sizeof(int), cudaMemcpyDeviceToHost));
    // apparently you don't have to unbind surfaces.
    cudaErrorCheck(cudaFreeArray(gpuImage));
}

int main(int argc, char **argv) {
    int width, height;
    int* binaryImage;
    CPUBitmap *bitmap;
    DataBlock data;
    BMP output;
    struct arguments parsed_args;

    //initialize CUDA - outputs to stdout
    //findCudaDevice(argc, (const char **)argv);

      //Suppress EasyBMP warnings, as they go to stdout and are silly.
      SetEasyBMPwarningsOff();
      //Get arguments, parsed results are in struct parsed_args.
      if (!get_args(argc, argv, &parsed_args)) {
        exit(EXIT_FAILURE);
      }

      fprintf(stderr,"%s Starting...\n\n", argv[0]);
        BMP input;
        if (parsed_args.mode == NORMAL_MODE) {
          //source and results image filenames
          char *SampleImageFname = parsed_args.filename;

          char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

          if (pSampleImageFpath == NULL) {
            fprintf(stderr,"%s could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
            exit(EXIT_FAILURE);
            }

          fprintf(stderr,"===============================================\n");
          fprintf(stderr,"Loading image: %s...\n", pSampleImageFpath);
          bool result = input.ReadFromFile(pSampleImageFpath);
          if (result == false) {
              fprintf(stderr,"\nError: Image file not found or invalid!\n");
              exit(EXIT_FAILURE);
          }
          fprintf(stderr,"===============================================\n");
        }
        else {
          makeRandomBMP(&input,parsed_args.width,parsed_args.width);
        }
        width = input.TellWidth();
        height = input.TellHeight();
        output.SetSize(width,height);
        output.SetBitDepth(32); // RGBA

        bitmap = new CPUBitmap( width, height, &data );
        data.bitmap = bitmap;
        copyBMPtoBitmap(&input,bitmap);
        binaryImage = new int[width*height];
        bitmapToBinary(bitmap,binaryImage);

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
      printf("%s,%d,%f,%f\n",parsed_args.mode==NORMAL_MODE?"normal":"random",
       width*height,
       gpuTime,milliseconds);
    }

	//Colourise
    colourise(binaryImage,bitmap,width,height);

    copyBitmapToBMP(bitmap,&output);
    //binaryToBitmap(binaryImage,&bitmap);
    //copyBitmapToBMP(&bitmap,&output);
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
    //DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
    char outname [255];
    if (parsed_args.mode == NORMAL_MODE) sprintf(outname,"%s-ccl-gpu.bmp",parsed_args.filename);
    else sprintf(outname,"random-%dx%d-ccl-gpu.bmp",width,height);
    output.WriteToFile(outname);
    //bitmap.display_and_exit((void (*)(void*))anim_exit);
    if (parsed_args.visualise) {
      bitmap->display_and_exit((void (*)(void*))anim_exit);
    }
    delete[] binaryImage;
}
