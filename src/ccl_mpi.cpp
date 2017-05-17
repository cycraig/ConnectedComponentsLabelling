#include "common_ccl.h"
#include <mpi.h>

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
    int root = p;
    while (root != blockRows[root-1])
        root = blockRows[root-1];
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
    // make smaller label priority
    if (j < i) {
        blockRows[i-1] = j;
    } else {
        blockRows[j-1] = i;
    }
}

#include <unistd.h>

void label(int* blockRows, int rank, int rowsPerRank, int width, int height) {
    // initial labelling
    initial_labels(blockRows, rank, rowsPerRank, width, height);
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
  BMP input;
  struct arguments parsed_args;

	if (rank==0) {
	  if (!start(argc, argv,
	      width, height,
	      input,
	      parsed_args)) {
			MPI_Finalize();
			exit(EXIT_FAILURE);
	}


	  bitmap = new CPUBitmap( width, height, &data );
	  data.bitmap = bitmap;
	  copyBMPtoBitmap(&input,bitmap);
	  binaryImage = new int[(width)*(height)];
	  bitmapToBinary(bitmap,binaryImage);
	  output.SetSize(width,height);
	  output.SetBitDepth(32); // RGBA

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
			sendcounts[i] = width*height-(processes-1)*rowsPerRank*width;
		}
		displs[i] = i*rowsPerRank*width;
	}
	MPI_Scatterv(binaryImage, sendcounts, displs, MPI_INT, blockRows, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

	if(rank == 0) fprintf(stderr,"LABELLING...\n");
    double start = MPI_Wtime();
	label(blockRows, rank, rowsPerRank, width, height);

	// gather row blocks (gatherv because last rank may have slack)
	MPI_Gatherv(blockRows, sendcounts[rank], MPI_INT, binaryImage, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // resolve final labels
    if(rank == 0) {
        merge(binaryImage, rowsPerRank, processes, width, height);
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
					printf("%s,%d,%f,%f\n",parsed_args.mode==NORMAL_MODE?"normal":"random",
					 width*height,
				   (stop-start)*1000.0,(stop_total-start_total)*1000.0);
				}
    }

	if(rank == 0) {
		finish(width, height,
						output,
						bitmap,
						binaryImage,
						parsed_args);
		delete[] binaryImage;
	}

	// clean up memory
	delete[] dims;
	delete[] blockRows;
	delete[] sendcounts;
	delete[] displs;

	MPI_Finalize();
}
