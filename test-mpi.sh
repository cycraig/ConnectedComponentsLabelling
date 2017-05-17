#/bin/bash
p=$1
while [ $p -lt 33 ]; do ./run_cluster-mpi.sh $p 1024 ; p=$(($p+1)); sleep 1; done

