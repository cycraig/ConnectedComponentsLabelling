#/bin/bash
w=$2
while [ $w -lt 2025 ]; do ./run_cluster.sh $1 $w ; w=$(($w+20)); sleep 1; done

