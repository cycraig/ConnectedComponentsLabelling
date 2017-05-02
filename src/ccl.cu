#include <cuda_runtime.h>
#include <cuda.h>
#include "../inc/helper_cuda.h"
#include "common_ccl.h"

void colourise(int* input, CPUBitmap* output, int** labelColours) {
	unsigned char *rgbaPixels = output->get_ptr();
	int width = output->x;
	int height = output->y;
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

			int r = labelColours[label][0];
			int g = labelColours[label][1];
			int b = labelColours[label][2];
			rgbaPixels[y*4*width+4*x]   = r;
			rgbaPixels[y*4*width+4*x+1] = g;
			rgbaPixels[y*4*width+4*x+2] = b;
			rgbaPixels[y*4*width+4*x+3] = 255;
		}
	}
}

void label(int* region, CPUBitmap* output, int width, int height) {
	// assume maximum number of labels
	int size = width*height/2.0+1;
	
	int** equivalenceMatrix = new int*[size];
	for(int i = 0; i < size; i++) {
		equivalenceMatrix[i] = new int[size];
		memset(equivalenceMatrix[i], 0, sizeof(int)*size);
		// reflexivity
		equivalenceMatrix[i][i] = 1;
	}

	int labelCount = 0;

	// initial labelling
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			
			// ignore background pixel
			if(region[y*width+x] == 0) continue;

			// check 4-nbrs
			int n=0,nw=0,ne=0,w=0,label=0;
			if(x > 0) {
				w = region[y*width+x-1];
				if(w) {
					label = w;
				}
			}
			if(y > 0) {
				n = region[(y-1)*width+x];
				if(n) {
					label = n;
					if(w) markEquivalent(equivalenceMatrix,n,w);
				}
			}
			if(y > 0 && x > 0) {
				nw = region[(y-1)*width+(x-1)];
				if(nw) {
					label = nw;
					if(w) markEquivalent(equivalenceMatrix,nw,w);
					if(n) markEquivalent(equivalenceMatrix,nw,n);
				}
			}
			if(y > 0 && x < (width-1)) {
				ne = region[(y-1)*width+(x+1)];
				if(ne) {
					label = ne;
					if(w)  markEquivalent(equivalenceMatrix,ne,w);
					if(n)  markEquivalent(equivalenceMatrix,ne,n);
					if(nw) markEquivalent(equivalenceMatrix,ne,nw);
				}
			}
			if(label == 0) {
				labelCount++;
				label = labelCount;
			}
			region[y*width+x] = label;
		}
	}
	resolveEquivalences(equivalenceMatrix,labelCount);
	int* labelArray = new int[labelCount+1];
	updateLabelArray(labelArray,equivalenceMatrix,labelCount);
	//printf("BEFORE:\n");
	//printLabels(region,width,height);
	updateRegion(region,labelArray,width,height);
	//printf("AFTER:\n");
	//printLabels(region,width,height);
	
	int** labelColours = new int*[labelCount+1];
	for(int i = 1; i < labelCount+1; i++) labelColours[i] = new int[3];
	getLabelColours(labelColours,labelCount);
	colourise(region,output,labelColours);

    // clean up memory
	for(int i = 0; i < size; i++)
		delete []equivalenceMatrix[i];
    delete []equivalenceMatrix;
    delete []labelArray;
	for(int i = 1; i < labelCount+1; i++) 
        delete []labelColours[i];
    delete []labelColours;
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

	printf("===================================\n");
    printf("Loading image: %s... \n", pSampleImageFpath);
    bool result = input.ReadFromFile(pSampleImageFpath);
	if (result == false) {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
        return 1;
    }
    printf("===================================\n");

	int width = input.TellWidth();
	int height = input.TellHeight();
	output.SetSize(width,height);
	output.SetBitDepth(32); // RGBA
    DataBlock   data;
    CPUBitmap bitmap( width, height, &data );
    data.bitmap = &bitmap;
    //HANDLE_ERROR( cudaEventCreate( &data.start ) );
    //HANDLE_ERROR( cudaEventCreate( &data.stop ) );
    copyBMPtoBitmap(&input,&bitmap);
    int* binaryImage = new int[width*height];
    bitmapToBinary(&bitmap,binaryImage);

    printf("LABELLING...\n");
    //double start_time = omp_get_wtime();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    label(binaryImage,&bitmap,width,height);

    //double end_time = omp_get_wtime();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("FINISHED...\n");
    //printf("Time elapsed: %f ms\n",(end_time-start_time)*1000.0);
    printf("Time elapsed (total): %.6f ms\n",milliseconds);

    copyBitmapToBMP(&bitmap,&output);
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
    //DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
    output.WriteToFile("out.bmp");
    bitmap.display_and_exit((void (*)(void*))anim_exit);
    delete[] binaryImage;
}

