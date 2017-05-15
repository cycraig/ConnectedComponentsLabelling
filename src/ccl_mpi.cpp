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
    // Initial Labelling
    int idx;
	for(int y = 0; y < rowsPerRank && (y+rank*rowsPerRank) < height; y++) {
        for(int x = 0; x < width; x++) {
            idx = y*width+x;
            if(blockRows[idx] > 0) {
		        // +1 to avoid 0 labels
		        blockRows[idx] = idx+1;
            }

        }
    }
}
// Return the root label of pixel p
int find(int p, int* blockRows) {
    //printf("Looking for root label of label %d...\n",p);
    int root = p;
    while (root != blockRows[root-1])
        root = blockRows[root-1];
    //printf("root = %d\n",root);
    while (p != root) {
        int newp = blockRows[p-1];
        blockRows[p-1] = root;
        p = newp;
    }
    return root;
}

// Replace sets containing x and y with their union.
void merge(int a, int b, int* blockRows) {
    int i = find(a,blockRows);
    int j = find(b,blockRows);
    if (i == j) return;
    //printf("(a=%d) %d equivalent to (b=%d) %d\n",a,i,b,j);
    // make smaller label priority
    if (j < i) {
        blockRows[i-1] = j;
    } else {
        blockRows[j-1] = i;
    }
}

#include <unistd.h>

void label(int* blockRows, int rank, int rowsPerRank, int width, int height) {
    /*if(rank == 1) {
        sleep(1);
    }*/
    // initial labelling
    //printf("Initial labelling before:\n");
    //printMatrix(blockRows,width,rowsPerRank);
    initial_labels(blockRows, rank, rowsPerRank, width, height);
    //printf("Initial labelling after:\n");
    //printMatrix(blockRows,width,rowsPerRank);*/
    int offset = rank*rowsPerRank*width;

    // union equivalent regions
	for(int y = 0; y < rowsPerRank && y+rowsPerRank*rank < height; y++) {
		for(int x = 0; x < width; x++) {
			// ignore background pixel
			if(blockRows[y*width+x] > 0) {
			    // check neighbour mask
			    int n=0,nw=0,ne=0,w=0,label=0;
                label = y*width+x+1;
			    if(x > 0) {
				    w = blockRows[y*width+x-1];
                    if(w != 0)
                        merge(label,w,blockRows);
			    }
			    if(y > 0) {
				    n = blockRows[(y-1)*width+x];
                    if(n != 0)
                        merge(label,n,blockRows);
			    }
			    if(y > 0 && x > 0) {
				    nw = blockRows[(y-1)*width+(x-1)];
                    if(nw != 0)
                        merge(label,nw,blockRows);
			    }
			    if(y > 0 && x < (width-1)) {
				    ne = blockRows[(y-1)*width+(x+1)];
                    if(ne != 0)
                        merge(label,ne,blockRows);
			    }
			    //image[y*width+x] = label;
            }
		}
	}

    // local labelling
	for(int y = 0; y < rowsPerRank && y+rowsPerRank*rank < height; y++) {
		for(int x = 0; x < width; x++) {
            if(blockRows[y*width+x] > 0)
    			blockRows[y*width+x] = find(blockRows[y*width+x],blockRows);
		}
	}
    // global labelling
    for(int y = 0; y < rowsPerRank && y+rowsPerRank*rank < height; y++) {
		for(int x = 0; x < width; x++) {
            if(blockRows[y*width+x] > 0)
    			blockRows[y*width+x] += offset;
		}
	}
}

void merge(int* unionImage, int rowsPerRank, int processes, int width, int height) {
    // union equivalent regions
    for(int rank = 1; rank < processes; rank++) {
        // look at row connecting adjacent region
        int y = rank*rowsPerRank;
        //printf("Checking row %d for rank %d\n",y,rank);
        for(int x = 0; x < width; x++) {
			// ignore background pixel
			if(unionImage[y*width+x] > 0) {
			    // check lower row
			    int n=0,nw=0,ne=0,label=0;
                label = y*width+x+1;
			    if(y > 0) {
				    n = unionImage[(y-1)*width+x];
                    if(n != 0)
                        merge(label,n,unionImage);
			    }
			    if(y > 0 && x > 0) {
				    nw = unionImage[(y-1)*width+(x-1)];
                    if(nw != 0)
                        merge(label,nw,unionImage);
			    }
			    if(y > 0 && x < (width-1)) {
				    ne = unionImage[(y-1)*width+(x+1)];
                    if(ne != 0)
                        merge(label,ne,unionImage);
			    }
			    //image[y*width+x] = label;
            }
		}
    }

    // final labelling
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
            if(unionImage[y*width+x] > 0)
    			unionImage[y*width+x] = find(unionImage[y*width+x],unionImage);
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
	struct arguments parsed_args;
	if(rank == 0) {
		//Suppress EasyBMP warnings, as they go to stdout and are silly.
		SetEasyBMPwarningsOff();
		//Get arguments, parsed results are in struct parsed_args.
		if (!get_args(argc, argv, &parsed_args)) {
			MPI_Finalize();
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
				MPI_Finalize();
				exit(EXIT_FAILURE);
		    }

			fprintf(stderr,"===============================================\n");
			fprintf(stderr,"Loading image: %s...\n", pSampleImageFpath);
			bool result = input.ReadFromFile(pSampleImageFpath);
			if (result == false) {
			    fprintf(stderr,"\nError: Image file not found or invalid!\n");
				MPI_Finalize();
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

    if(rank == 0) {
        //printMatrix(binaryImage, width, height);
    }

	if(rank == 0) fprintf(stderr,"LABELLING...\n");
    double start = MPI_Wtime();
	label(blockRows, rank, rowsPerRank, width, height);

	// gather row blocks (gatherv because last rank may have slack)
	MPI_Gatherv(blockRows, sendcounts[rank], MPI_INT, binaryImage, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        //printMatrix(binaryImage, width, height);
    }

    // resolve final labels
    if(rank == 0) {
        //printf("MERGING FINAL IMAGE...\n");
        merge(binaryImage, rowsPerRank, processes, width, height);
    }

    if(rank == 0) {
        //printMatrix(binaryImage, width, height);
    }

    double stop = MPI_Wtime();

    double stop_total = MPI_Wtime();
    if(rank == 0) {
        fprintf(stderr,"FINISHED...\n");
				if (!parsed_args.bench) {
        	printf("Time elapsed (labelling): %.6f ms\n",(stop-start)*1000.0);
        	printf("Time elapsed (total):     %.6f ms\n",(stop_total-start_total)*1000.0);
				}
				else {
					printf("%s,%d,%d,%f,%f\n",parsed_args.mode==NORMAL_MODE?"normal":"random",
					 width, height,
				   (stop-start)*1000.0,(stop_total-start_total)*1000.0);
				}
    }

    if(rank == 0) {
        //Colourise
        fprintf(stderr,"Colouring image...\n");
        colourise(binaryImage,bitmap,width,height);
        fprintf(stderr,"Done colouring...\n");
    }

	if(rank == 0) {
		//imageToBitmap(binaryImage,&bitmap);
		//binaryToBitmap(binaryImage,bitmap);
		copyBitmapToBMP(bitmap,&output);
		//binaryToBitmap(binaryImage,&bitmap);
		//copyBitmapToBMP(&bitmap,&output);
		//HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
		//DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
		char outname [255];
		if (parsed_args.mode == NORMAL_MODE) sprintf(outname,"%s-ccl-mpi.bmp",parsed_args.filename);
		else sprintf(outname,"random-%dx%d-ccl-mpi.bmp",width,height);
		output.WriteToFile(outname);
		//bitmap.display_and_exit((void (*)(void*))anim_exit);
		if (parsed_args.visualise) {
			bitmap->display_and_exit((void (*)(void*))anim_exit);
		}
		delete[] binaryImage;
	}

	// clean up memory
	delete[] dims;
	delete[] blockRows;
	delete[] sendcounts;
	delete[] displs;

	MPI_Finalize();
}
