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

void label(int* blockRows, int width, int height) {
	UF unionFind(height*width+1);
    // marking equivalences

	for(int y = 0; y < height; y++) {
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
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
            if(blockRows[y*width+x] > 0)
    			blockRows[y*width+x] = unionFind.find(y*width+x+1);
		}
	}
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


  bitmap = new CPUBitmap( width, height, &data );
  data.bitmap = bitmap;
  copyBMPtoBitmap(&input,bitmap);
  binaryImage = new int[(width)*(height)];
  bitmapToBinary(bitmap,binaryImage);
  output.SetSize(width,height);
  output.SetBitDepth(32); // RGBA

	fprintf(stderr,"LABELLING...\n");
  double start = MPI_Wtime();
	label(binaryImage, width, height);

  double stop = MPI_Wtime();
      if (!parsed_args.bench) {
        printf("Time elapsed (total):     %.6f ms\n",(stop-start)*1000.0);
      }
      else {
        printf("%s,%d,%d,%f\n",parsed_args.mode==NORMAL_MODE?"normal":"random",
         width, height,
         (stop-start)*1000.0);
      }

  finish(width, height,
          output,
          bitmap,
          binaryImage,
          parsed_args);
  delete[] binaryImage;
  return 0;
}
