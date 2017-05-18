#/bin/bash
p=1
w=$1
r=2

make
#while [ $w -lt 2235 ]; do

while [ $r -lt 33 ]; do
	echo "(GPU) Submit job for p=$p, w=$w, r=$r"
	./run_cluster-gpu.sh $p $w $r
        ./run_cluster-gpu_global.sh $p $w $r
	r=$(($r * 2))
done

echo "(Union) Submit job for p=$p, w=$w"
./run_cluster-union.sh $p $w

p=2
while [ $p -lt 33 ]; do
        echo "(MPI) Submit job for p=$p, w=$w"
        ./run_cluster-mpi.sh $p $w
        p=$(($p * 2))
done

#sleep 1
#w=$(($w + 100))

#done
