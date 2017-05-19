# HPC_CCL
Project for COMS4040A: High Performance Computing

#Make instructions
Simply run make. Make sure your nvcc executables are exported properly!

#Folder description:
./data : Contains test images of varying sizes.
./results : Contains subdirectories which have csv files from benchmarking. Graphs are in the subdirectories.
./documents : Reference papers used for the algorithms and mentioned in report.
./inc : Neccesary libraries
./presentation: Our presentation
./samples : Additional reference code
./report : Contains our report
,/test-scripts : Scripts used to perform benchmarks by submitting jobs on the cluster. These were hacked together, so we cannot guarantee that they will work 100% of the time.
./src: Our source code.
  -> ./src/old : Old serial implementations. Need not be regarded.
  -> ./src/ccl_gpu.cu : Our CUDA surface implementation.
  -> ./src/ccl_gpu_lobal.cu : Our global memory CUDA implementation.
  -> ./src/ccl_mpi.cpp : Our MPI implementation.
  -> ./src/ccl_serial.cpp : Our serial implementation.
  -> ./src/common_ccl.cpp ; .h : Commmon functions
  -> ./src/EasyBMP* : EasyBMP library files (not our code)

RELEVANT FOR MARKING: ./src ./documents ./report ./presentation

#Usage
First run make

To run a test image through, say, ccl_serial:
./ccl_serial -f f0001.bmp

Note that the program ONLY ACCEPTS 24 BIT RGB FILES. Some test images are provided in the data folder. It is not neccesary to give the full path, only the filename.
This will save the coloured image to the current directory as f0001.bmp-ccl_serial.bmp.

To use visualisations, use the -v flag ,e.g:
./ccl_serial -f f0001.bmp -v

This will bring up a visualisation window before saving the image.

To generate a psuedo-random image, you can use, e.g.:
./ccl_serial -m random -w 1024

Note that it is mandatory to specify -w when -m is set to random.
This will save the image as random-1024x1024-ccl_serial.bmp

For a full argument description, Try --help on any of the executables.
