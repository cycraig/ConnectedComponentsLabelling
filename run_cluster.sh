#!/bin/sh

if [ ! $# -eq 2 ]
then
	echo "Usage:   $0 num_processes random_image_size"
	echo "Example: $0 8 1024"
	exit
fi
p=$1
f=$2
nodes=$((($p+3-1)/3));
jobname="mpi_ccl_${p}_${w}"
echo "Starting job ${jobname}..."
dir=$(pwd)
echo "Requesting $p processes over $nodes node/s"
cat <<EOS | qsub -
#!/bin/bash
#PBS -N $jobname
#PBS -e job.err
#PBS -o job.log
#PBS -l walltime=0:01:00
#PBS -q batch
#PBS -l nodes=$nodes:ppn=3
cd $dir
make
./ccl_gpu_global -f "$f" -b >> "new-gpu_global${f}.out"
./ccl_gpu -f "$f" -b >> "new-gpu${f}.out"
./ccl_unionfind -f "$f" -b >> "new-union${f}.out"
#./ccl_mpi -f "$f" -b >> "new-mpi${f}.out"
EOS
