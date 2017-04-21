
#include <cuda_runtime.h>
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <helper_cuda.h>
#include <cstring> 
#include <cmath>
#include "cuda.h"
#include "book.h" //HANDLE_ERROR
#include "cpu_bitmap.h" 

#include "EasyBMP.h"

int regionWidth = 55;
int regionHeight = 55;
int total_index;

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

void colourise(int* input, CPUBitmap* output, int width, int height, int** labelColours) {
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

void getLabelColours(int** labelColours, int maxLabels) {
    //printf("Maximum %d label colours\n",maxLabels);
	for(int i = 1; i <= maxLabels; i++) {
		labelColours[i][0] = i * 131 % 255;
		labelColours[i][1] = i * 241 % 255;
		labelColours[i][2] = i * 251 % 255;
	}
}

void markEquivalent(int** equivalenceMatrix, int a, int b) {
	equivalenceMatrix[a][b] = 1;
	equivalenceMatrix[b][a] = 1;
}

void printMatrix(int** matrix, int width, int height) {
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int val = matrix[y][x];
			if(val < 10) {
				printf(" %d ",val);
			} else {
				printf("%d ",val);
			}
		}
		printf("\n");
	}
}

void printArray(int* array, int size) {
	for(int i = 0; i < size; i++) {
		printf("%d ",array[i]);
	}
	printf("\n");
}

void updateLabelArray(int* labelArray, int** L, int maxLabel) {
	labelArray[0] = 0;
	for(int label = 1; label <= maxLabel; label++) {
		for(int i = 1; i <= maxLabel; i++) {
			if(L[label][i]) {
				labelArray[label] = i;
				break;
			}
		}
	}
	//printf("LABEL ARRAY:\n");
	//printArray(labelArray,maxLabel+1);
}

void resolveEquivalences(int** L, int maxLabel) {
	int n = maxLabel;
	for(int j = 1; j <= n; j++) {
		for(int i = 1; i <= n; i++) {
			if(L[i][j] == 1) {
				for(int k = 1; k <= n; k++) {
					L[i][k] = L[i][k] || L[j][k];
				}
			}
		}
	}
	//printf("EQUIVALENCE MATRIX:\n");
	//printMatrix(L,n+1,n+1);
}

void updateRegion(int* region, int* labelArray, int width, int height) {
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int idx = y*width+x;
			if(region[idx] > 0) {
				region[idx] = labelArray[region[idx]];
			}
		}
	}
}

void printLabels(int* region, int width, int height) {
	for(int y = 0; y < height; y++) {
		for(int x = 0; x < width; x++) {
			int val = region[y*width+x];
			if(val < 10) {
				printf(" %d ",val);
			} else {
				printf("%d ",val);
			}
		}
		printf("\n");
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
	colourise(region,output, width, height, labelColours);
}

#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)

void resolveEquivalence(int p, int q, int pregion, int qregion, int** label_list, int* label_count) {
    // step 1
    int index1 = label_list[qregion][q];
    int index2 = label_list[pregion][p];
    //printf("Global labels %d (local %d) and %d (local %d) are equivalent for regions %d and %d\n",index1,q,index2,p,qregion,pregion);
    if(index1 != index2) {
        // step 2
        int small = min(index1,index2);
        int large = max(index1,index2);
        
        for(int k = 0; k <= pregion; k++) {
            for(int j = 1; j <= label_count[k]; j++) {
                if(label_list[k][j] > large) {
                    label_list[k][j] = label_list[k][j] - 1;
                } else if(label_list[k][j] == large) {
                    label_list[k][j] = small;
                }
            }
        }
        total_index = total_index - 1;

        //printf("-- updated label_list:\n");
        //for(int i = 0; i <= pregion; i++)
        //   printArray(label_list[i],label_count[i]+1);
        //printf("\n");
    }
}

unsigned int getRegion(int y, int x, int imageWidth, int imageHeight) {
    unsigned int regions_across = (imageWidth-1)/regionWidth+1;
    unsigned int regionx = x/regionWidth;
    unsigned int regiony = y/regionHeight;
    //printf("(x,y)=(%d,%d) is in region %d\n",x,y,(regiony*regions_across+regionx));
    return (regiony*regions_across+regionx);
}

void merge_region(int* image, int width, int height, int region, int** label_list, int* label_count) {
    unsigned int regions_across = (width-1)/regionWidth+1;
    unsigned int regions_down = (height-1)/regionHeight+1;
    unsigned int regionx = (region%regions_across);
    unsigned int regiony = (region/regions_across);
    unsigned int startx = regionx*regionWidth;
    unsigned int starty = regiony*regionHeight;
    unsigned int regionSize = regionWidth*regionHeight;

    // step 1 (corner pixel)
    int x = startx;
    int y = starty;
    int p = image[y*width + x];
    int qind,qregion;
    //printf("Merging region %d\n",region);
    if(p > 0) {
        // check 8-nbrs
        //printf("Checking merge neighbours for corner pixel at (x,y)=(%d,%d): %d\n",x,y,p);
        
        // south-west
        qind = (y+1)*width+x-1;
        if(x > 0 && y < (height-1) && image[qind] > 0) {
            qregion = getRegion(y+1,x-1,width,height);
            //printf("SW - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y+1,qregion);
			resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		}

        // west
        qind = y*width+x-1;
		if(x > 0 && image[qind] > 0) {
            qregion = getRegion(y,x-1,width,height);
            //printf("W  - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y,qregion);
			resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		}

        // north-west
        qind = (y-1)*width+x-1;
		if(y > 0 && x > 0 && image[qind] > 0) {
			qregion = getRegion(y-1,x-1,width,height);
            //printf("NW - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y-1,qregion);
			resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		}

        // north
        qind = (y-1)*width+x;
		if(y > 0 && image[qind] > 0) {
			qregion = getRegion(y-1,x,width,height);
            //printf("N  - Equivalent to (x,y)=(%d,%d) in region %d\n",x,y-1,qregion);
			resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		}
		
        // north-east
        qind = (y-1)*width+x+1;
		if(y > 0 && x < (width-1) && image[qind] > 0) {
			qregion = getRegion(y-1,x+1,width,height);
            //printf("NE - Equivalent to (x,y)=(%d,%d) in region %d\n",x+1,y-1,qregion);
			resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		}
    }

    // step 2 (first column pixels)
    x = startx;
    for(y = starty+1; y < (regiony+1)*regionHeight && y < height; y++) {
        p = image[y*width + x];
        if(p > 0) {
            //printf("Checking merge neighbours for column pixel at (x,y)=(%d,%d): %d\n",x,y,p);

            // north-west
            qind = (y-1)*width+x-1;
		    if(y > 0 && x > 0 && image[qind] > 0) {
			    qregion = getRegion(y-1,x-1,width,height);
                //printf("NW - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y-1,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }

            // west
            qind = y*width+x-1;
		    if(x > 0 && image[qind] > 0) {
                qregion = getRegion(y,x-1,width,height);
                //printf("W  - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }
            
            // south-west
            qind = (y+1)*width+x-1;
            // don't check past bottom of region
            if(y < ((regiony+1)*regionHeight-1) && x > 0 && y < (height-1) && image[qind] > 0) {
                qregion = getRegion(y+1,x-1,width,height);
                //printf("SW - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y+1,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }
        }
    }

    // step 3 (first row pixels)
    y = starty;
    for(x = startx+1; x < (regionx+1)*regionWidth && x < width; x++) {
        p = image[y*width + x];
        if(p > 0) {
            //printf("Checking merge neighbours for row pixel at (x,y)=(%d,%d): %d\n",x,y,p);

            // north-west
            qind = (y-1)*width+x-1;
		    if(y > 0 && x > 0 && image[qind] > 0) {
			    qregion = getRegion(y-1,x-1,width,height);
                //printf("NW - Equivalent to (x,y)=(%d,%d) in region %d\n",x-1,y-1,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }

            // north
            qind = (y-1)*width+x;
		    if(y > 0 && image[qind] > 0) {
			    qregion = getRegion(y-1,x,width,height);
                //printf("N  - Equivalent to (x,y)=(%d,%d) in region %d\n",x,y-1,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }
		
            // north-east
            qind = (y-1)*width+x+1;
            // don't check past right of region
		    if(x < ((regionx+1)*regionWidth-1) && y > 0 && x < (width-1) && image[qind] > 0) {
			    qregion = getRegion(y-1,x+1,width,height);
                //printf("NE - Equivalent to (x,y)=(%d,%d) in region %d\n",x+1,y-1,qregion);
			    resolveEquivalence(p,image[qind],region,qregion,label_list,label_count);
		    }
        }
    }
}

void printRegion(int* image,int width, int height, int region) {
    unsigned int regions_across = (width-1)/regionWidth+1;
    unsigned int regions_down = (height-1)/regionHeight+1;
    unsigned int regionx = (region%regions_across);
    unsigned int regiony = (region/regions_across);
    unsigned int startx = regionx*regionWidth;
    unsigned int starty = regiony*regionHeight;
    //printf("startx = %d\nstarty = %d\n",startx,starty);
    for(int y = starty; y < height && y < (regiony+1)*regionHeight; y++) {
		for(int x = startx; x < width && x < (regionx+1)*regionWidth; x++) {
			int val = image[y*width+x];
			if(val < 10) {
				printf(" %d ",val);
			} else {
				printf("%d ",val);
			}
		}
		printf("\n");
	}
}


void label_region(int* image, int width, int height, int region, int** label_list, int* label_count) {
    // step i
   
    // assume maximum number of labels
	int size = regionWidth*regionHeight/2.0+1;
	
    unsigned int regions_across = (width-1)/regionWidth+1;
    unsigned int regions_down = (height-1)/regionHeight+1;
    unsigned int regionx = (region%regions_across);
    unsigned int regiony = (region/regions_across);
    unsigned int startx = regionx*regionWidth;
    unsigned int starty = regiony*regionHeight;

	int** equivalenceMatrix = new int*[size];
	for(int i = 0; i < size; i++) {
		equivalenceMatrix[i] = new int[size];
		memset(equivalenceMatrix[i], 0, sizeof(int)*size);
		// reflexivity
		equivalenceMatrix[i][i] = 1;
	}

	int labelCount = 0;

    //printf("Region %d:\n startx = %d; starty = %d;\n",region,startx,starty); 

	// initial labelling
	for(int y = starty; y < (regiony+1)*regionHeight && y < height; y++) {
		for(int x = startx; x < (regionx+1)*regionWidth && x < width; x++) {
			
			// ignore background pixel
			if(image[y*width+x] > 0) {

			    // check 4-nbrs
			    int n=0,nw=0,ne=0,w=0,label=0;
			    if(x > startx) {
				    w = image[y*width+x-1];
				    if(w) {
					    label = w;
				    }
			    }
			    if(y > starty) {
				    n = image[(y-1)*width+x];
				    if(n) {
					    label = n;
					    if(w) markEquivalent(equivalenceMatrix,n,w);
				    }
			    }
			    if(y > starty && x > startx) {
				    nw = image[(y-1)*width+(x-1)];
				    if(nw) {
					    label = nw;
					    if(w) markEquivalent(equivalenceMatrix,nw,w);
					    if(n) markEquivalent(equivalenceMatrix,nw,n);
				    }
			    }
			    if(y > starty && x < ((regionx+1)*regionWidth-1) && x < (width-1)) {
				    ne = image[(y-1)*width+(x+1)];
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
			    image[y*width+x] = label;
            }
		}
	}

    // step ii
    //printf("Label count for region %d = %d\n",region,labelCount);
    label_count[region] = labelCount;
    label_list[region] = new int[labelCount+1];
    
    // step iii
    resolveEquivalences(equivalenceMatrix,labelCount);
	updateLabelArray(label_list[region],equivalenceMatrix,labelCount);
    
    // step iv
    int maxLabel = 0;
    for(int j = 1; j <= labelCount; j++) {
        if(label_list[region][j] > maxLabel) maxLabel = label_list[region][j];
        label_list[region][j] = label_list[region][j] + total_index;
    }

    //printf("label_list[%d]:\n",region);
    //printArray(label_list[region],label_count[region]+1);
    
    // step v
    total_index = total_index + maxLabel;

    // step vi
    if(region > 0) {
        merge_region(image, width, height, region, label_list, label_count);
    }

    //printf("label_list after merging region %d:\n",region);
    //for(int i = 0; i <= region; i++)
    //    printArray(label_list[i],label_count[i]+1);
}

void updateRegionToGlobalLabels(int* image, int width, int height, int region, int** label_list) {
    unsigned int regions_across = (width-1)/regionWidth+1;
    unsigned int regions_down = (height-1)/regionHeight+1;
    unsigned int regionx = (region%regions_across);
    unsigned int regiony = (region/regions_across);
    unsigned int startx = regionx*regionWidth;
    unsigned int starty = regiony*regionHeight;
    
    for(int y = starty; y < height && y < (regiony+1)*regionHeight; y++) {
		for(int x = startx; x < width && x < (regionx+1)*regionWidth; x++) {
			int idx = y*width+x;
			if(image[idx] > 0) {
				image[idx] = label_list[region][image[idx]];
			}
		}
	}
}

void fast_label(int* image, CPUBitmap* output, int width, int height) {
    // Step 1
    //printf("Step 1...\n");
    total_index = 0;
    int numRegions = ((width-1)/regionWidth+1)*((height-1)/regionHeight+1);

    // Step 2
    //printf("Step 2...\n");
    int** label_list = new int*[numRegions];
    int* label_count = new int[numRegions];
    for(int i = 0; i < numRegions; i++) {
        //printf("Labelling region %d:\n",i);
        //printRegion(image,width,height,i);
        label_region(image, width, height, i, label_list, label_count);
        //printRegion(image,width,height,i);
    }

    //printf("BEFORE:\n");
	//printLabels(image,width,height);

    // Step 3
    //printf("Step 3...\n");
    for(int i = 0; i < numRegions; i++) {
        //printf("label_list[%d]:\n",i);
	    //printArray(label_list[i],label_count[i]+1);
        updateRegionToGlobalLabels(image,width,height,i,label_list);   
    }

	//printf("AFTER:\n");
	//printLabels(image,width,height);
	
    // Colourise
	int** labelColours = new int*[total_index+1];
	for(int i = 1; i <= total_index; i++) labelColours[i] = new int[3];
	getLabelColours(labelColours,total_index);
	colourise(image,output,width,height,labelColours);
}

int main(int argc, char **argv) {
    printf("%s Starting...\n\n", argv[0]);

    //initialize CUDA
    findCudaDevice(argc, (const char **)argv);

    //source and results image filenames
    char SampleImageFname[] = "test.bmp";
    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

    if (pSampleImageFpath == NULL) {
        printf("%s could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    BMP input;
    BMP output;

	printf("===================================\n");
    printf("Loading image: %s... ", pSampleImageFpath);
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
    data.totalTime = 0;
    data.frames = 0;
    //HANDLE_ERROR( cudaEventCreate( &data.start ) );
    //HANDLE_ERROR( cudaEventCreate( &data.stop ) );
    copyBMPtoBitmap(&input,&bitmap);
    int* binaryImage = new int[width*height];
    bitmapToBinary(&bitmap,binaryImage);

    //printf("PIXEL (99,77) = %d\n",binaryImage[77*width+99]);
    //printf("PIXEL (98,78) = %d\n",binaryImage[78*width+98]);
    //printf("PIXEL (97,79) = %d\n",binaryImage[79*width+97]);
    

    printf("LABELLING...\n");
    fast_label(binaryImage,&bitmap,width,height);
    printf("FINISHED...\n");

    //printf("PIXEL (99,77) = %d\n",binaryImage[77*width+99]);
    //printf("PIXEL (98,78) = %d\n",binaryImage[78*width+98]);
    //printf("PIXEL (97,79) = %d\n",binaryImage[79*width+97]);

    copyBitmapToBMP(&bitmap,&output);
    //binaryToBitmap(binaryImage,&bitmap);
    //copyBitmapToBMP(&bitmap,&output);
    //HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), ImgSrc, imageSize, cudaMemcpyHostToHost ) );
    //DumpBmpAsGray("out.bmp", ImgSrc, ImgStride, ImgSize);
    output.WriteToFile("out.bmp");
    bitmap.display_and_exit((void (*)(void*))anim_exit);
}

