#include "common_ccl.h"
#include <mpi.h>
#include <omp.h>

int commSize;

class UF {
    int *id;
public:
    // Create an empty union find data structure with N isolated sets.
    UF(int N) {
        id = new int[N];
        for (int i = 0; i<N; i++) {
            id[i] = i;
        }
    }

    ~UF() { 
        delete[] id; 
    }

    // Return the id of component corresponding to object p.
    int find(int p) {
        int root = p;
        while (root != id[root])
            root = id[root];
        while (p != root) { 
            int newp = id[p];
            id[p] = root;
            p = newp;
        }
        return root;
    }
    
    // Replace sets containing x and y with their union.
    void merge(int x, int y) {
        int i = find(x);
        int j = find(y);
        if (i == j) return;
        //printf("(a=%d) %d equivalent to (b=%d) %d\n",x,i,y,j);
        // make smaller label priority
        if (j < i) {
            id[i] = j;
        } else { 
            id[j] = i;
        }
    }
    
    // Are objects x and y in the same set?
    bool connected(int x, int y) { 
        return find(x) == find(y);
    }
};

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

void initial_labels(int* blockRows, int rank, int rowsPerRank, int width, int height) {
    // STEP 1 - Initial Labelling
    int startLabel = rank*rowsPerRank*width+1;
	int endIndex = rowsPerRank;
	for(int i = 0; i < endIndex; i++) {
		// +1 to avoid 0 labels
		//blockRows[i*n
    }
}

void label(int* blockRows, int rank, int rowsPerRank, int width, int height) {
	UF unionFind(rowsPerRank*width+1);

    int offset = rank*rowsPerRank*width;

    // marking equivalences
	
	for(int y = 0; y < rowsPerRank && y+rowsPerRank*rank < height; y++) {
		for(int x = 0; x < width; x++) {
			// ignore background pixel
			if(blockRows[y*width+x] > 0) {
			    // check neighbour mask
			    int n=0,nw=0,ne=0,w=0,label=0;
                // local label
                label = y*width+x+1;
			    if(x > 0) {
				    w = blockRows[y*width+x-1];
					if(w != 0) 
	                    unionFind.merge(label,y*width+x-1+1);
			    }
			    if(y > 0) {
				    n = blockRows[(y-1)*width+x];
					if(n != 0)
                   		unionFind.merge(label,(y-1)*width+x+1);
			    }
			    if(y > 0 && x > 0) {
				    nw = blockRows[(y-1)*width+(x-1)];
					if(nw != 0)
                    	unionFind.merge(label,(y-1)*width+(x-1)+1);
			    }
			    if(y > 0 && x < (width-1)) {
				    ne = blockRows[(y-1)*width+(x+1)];
					if(ne != 0)
                    	unionFind.merge(label,(y-1)*width+(x+1)+1);
			    }
			    //image[y*width+x] = label;
            }
		}
	}

    // initial labelling
	for(int y = 0; y < rowsPerRank && y+rowsPerRank*rank < height; y++) {
		for(int x = 0; x < width; x++) {
            if(blockRows[y*width+x] > 0)
    			blockRows[y*width+x] = unionFind.find(y*width+x+1);
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
		char SampleImageFname[] = "test.bmp";
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

    // start timing for overhead
    MPI_Barrier(MPI_COMM_WORLD);
    double start_total = MPI_Wtime();

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
	
    /*if(rank == 0) {
        printf("Received on rank %d, %d elements over %d rows\n",rank,sendcounts[rank],rowsPerRank);
        printMatrix(blockRows,width,rowsPerRank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1) {
        printf("Received on rank %d, %d elements over %d rows:\n",rank,sendcounts[rank],rowsPerRank);
        printMatrix(blockRows,width,rowsPerRank);
    }*/
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

	if(rank == 0) printf("LABELLING...\n");
    double start = MPI_Wtime();
    double start_time = omp_get_wtime();
	label(blockRows, rank, rowsPerRank, width, height);
	

	// gather row blocks (gatherv because last rank may have slack)
	MPI_Gatherv(blockRows, sendcounts[rank], MPI_INT, binaryImage, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // resolve final labels
    // TODO

    if(rank == 0) {
        //printMatrix(binaryImage, width, height);
    }
    
    if(rank == 0) {
        //Colourise
        printf("Colouring image...\n");
        colourise(binaryImage,bitmap,width,height);
        printf("Done colouring...\n");
    }
    double stop = MPI_Wtime();

    double stop_total = MPI_Wtime();
    if(rank == 0) {
        printf("FINISHED...\n");
        printf("Time elapsed (labelling): %.6f ms\n",(stop-start)*1000.0);
        printf("Time elapsed (total):     %.6f ms\n",(stop_total-start_total)*1000.0);
    }

    
    double end_time = omp_get_wtime();
    printf("Time elapsed: %f ms\n",(end_time-start_time)*1000.0);

    if(rank == 0) {
	    //imageToBitmap(binaryImage,&bitmap);
	    //binaryToBitmap(binaryImage,bitmap);
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

