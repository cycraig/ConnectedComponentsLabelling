#/bin/bash
w=$2
while [ $w -lt 33 ]; do ./run_cluster-gpu.sh $1 1024 $w ; w=$(($w+1)); sleep 1; done

