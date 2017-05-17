#!/bin/sh

if [ ! $# -eq 2 ]
then
	echo "Usage:   $0 num_processes random_image_size"
	echo "Example: $0 8 1024"
	exit
fi
p=$1
nodes=$((($p+3-1)/3));
w=$2
jobname="mpi_ccl_${p}_${w}"
echo "Starting job ${jobname}..."
dir=$(pwd)
echo "Requesting $p processes over $nodes node/s"
cat <<EOS | qsub -
#!/bin/sh
#PBS -N $jobname
#PBS -e job.err
#PBS -o job.log
#PBS -l walltime=0:01:00
#PBS -q batch
#PBS -l nodes=$nodes:ppn=3
cd $dir
mpirun -np $p ./ccl_mpi -m random -w $w -b >> "mpi${p}.out"
EOS
