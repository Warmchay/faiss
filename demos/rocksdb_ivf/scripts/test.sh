#!/usr/bin/bash

executor="/home/wq/code/faiss/demos/rocksdb_ivf/build/test_ivf"

# Vars
nlists=(100 500 1000 5000 10000 50000 100000)
probes=(10 20 40 80 160 320 640 1280 2560 5120 10240)

res_dir="/data1/wq/bigann/result/ivf_flat_sift50M/"
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