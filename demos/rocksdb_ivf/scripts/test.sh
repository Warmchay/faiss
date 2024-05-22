#!/usr/bin/bash

executor="/home/wq/code/faiss/demos/rocksdb_ivf/build/test_ivf"

# Vars
nlists=(50 100 500 1000 2500 5000)
# nlists=(10000)
probes=(1 2 4 8 16 32 64)

res_dir="/data1/wq/bigann/result/ivf_flat_sift100M/"
mkdir -p ${res_dir}
# db="/data1/wq/bigann/db"

for nlist in "${nlists[@]}"
do
    for probe in "${probes[@]}"
    do
        if [ $nlist -gt $probe ]
        then  
            # res_time_path="${res_dir}/${nlist}_time.txt"
            # res_recall_path="${res_dir}/${nlist}_recall.txt"

            # db_path="${db}/${nlist}_${probe}"
            # mkdir -p ${db_path}

            ${executor} --nlist $nlist \
                        --probes $probe
        fi
    done
done