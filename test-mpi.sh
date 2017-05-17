#/bin/bash
p=$1
while [ $w -lt 33 ]; do ./run_cluster-mpi.sh $p 1024 ; w=$(($w+1)); sleep 1; done

