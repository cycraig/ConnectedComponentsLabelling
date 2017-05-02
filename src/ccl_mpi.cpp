#include "common_ccl.h"
#include <mpi.h>

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

int main(int argc, char **argv) {
	int rank, processes;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	
	int width, height;
	int* dims = new int[2];
	int* binaryImage;
	CPUBitmap *bitmap;
	DataBlock data;
	BMP output;
	if(rank == 0) {
		printf("%s Starting...\n\n", argv[0]);

		//source and results image filenames
		char SampleImageFname[] = "3pixeldeath.bmp";
		char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

		if (pSampleImageFpath == NULL) {
			printf("%s could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
			MPI_Finalize();
			exit(EXIT_FAILURE);
	    }

		BMP input;

		printf("===============================================\n");
		printf("Loading image: %s...\n", pSampleImageFpath);
		bool result = input.ReadFromFile(pSampleImageFpath);
		if (result == false) {
		    printf("\nError: Image file not found or invalid!\n");
			MPI_Finalize();
		    exit(EXIT_FAILURE);
		}
		printf("===============================================\n");

		width = input.TellWidth();
		height = input.TellHeight();
		output.SetSize(width,height);
		output.SetBitDepth(32); // RGBA
		
		bitmap = new CPUBitmap( width, height, &data );
		data.bitmap = bitmap;
		//HANDLE_ERROR( cudaEventCreate( &data.start ) );
		//HANDLE_ERROR( cudaEventCreate( &data.stop ) );
		copyBMPtoBitmap(&input,bitmap);
		binaryImage = new int[width*height];
		bitmapToBinary(bitmap,binaryImage);

		dims[0] = width;
		dims[1] = height;
	}

	// broadcast image dimensions (in array to avoid an extra broadcast)
	MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);
	width = dims[0];
	height = dims[1];

	// distribute row blocks (scatterv because last rank may have slack)
	int rowsPerRank = (height-1)/processes+1; // ceil
	// calculate counts and offsets
	int* blockRows = new int[rowsPerRank*width];
	int* sendcounts = new int[processes];
	int* displs = new int[processes];
	for(int i = 0; i < processes; i++) {
		if(i < (processes-1)) {
			sendcounts[i] = rowsPerRank*width;
		} else {
			//sendcounts[i] = rowsPerRank*width - (processes*rowsPerRank*width - width*height);
			sendcounts[i] = width*height-(processes-1)*rowsPerRank*width;
		}
		displs[i] = i*rowsPerRank*width;
	}
	MPI_Scatterv(binaryImage, sendcounts, displs, MPI_INT, blockRows, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);
	

    /*printf("LABELLING...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

   	// double start_time = omp_get_wtime();
    gpu_label(binaryImage,&bitmap,width,height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("FINISHED...\n");
    //printf("Time elapsed: %f ms\n",(end_time-start_time)*1000.0);
    printf("Time elapsed (total): %.6f ms\n",milliseconds);*/

	// gather row blocks (gatherv because last rank may have slack)
	MPI_Gatherv(blockRows, sendcounts[rank], MPI_INT, binaryImage, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);


	if(rank == 0) {
		//imageToBitmap(binaryImage,&bitmap);
		binaryToBitmap(binaryImage,bitmap);
		copyBitmapToBMP(bitmap,&output);
		//binaryToBitmap(binaryImage,&bitmap);
		//copyBitmapToBMP(&bitmap,&output);
		//HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
		//DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
		output.WriteToFile("out.bmp");
		//bitmap.display_and_exit((void (*)(void*))anim_exit);
		delete[] binaryImage;
	}

	// clean up memory
	delete[] dims;
	delete[] blockRows;
	delete[] sendcounts;
	delete[] displs;

	MPI_Finalize();
}

