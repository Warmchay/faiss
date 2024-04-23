#!/usr/bin/bash

# Test for ivf flat in rocksdb
# Define dataset's vars
dim=128
nlists=(100 500 1000 5000 10000 50000 100000)

# Define use rocksdb or not
use_db=true
db_path="/data1/wq/bigann/db"

# Select used files
learn_file="/data1/wq/bigann/bigann_learn_without_header.bvecs"
base_file="/data1/wq/bigann/bigann_base_without_header.bvecs"
query_file="/data1/wq/bigann/bigann_query_without_header.bvecs"
gt_file="/data1/wq/bigann/gnd/idx_1M.ivecs"

# Define searching vars
probes=(10 20 40 80 160 320 640 1280 2560 5120 10240)

# Decide res files location
res_dir="/data1/wq/bigann/result/ivf_flat/sift1b"


# Define executor
executor="/home/wq/code/faiss/demos/rocksdb_ivf/build/rocksdb_ivf"

for nlist in $nlists
do
    nlist_dir="${res_dir}/${nlist}"
    mkdir -p $nlist_dir
    for probe in $probes
    do
        if [ $nlist -gt $probe ]
        then
            res_time_path="${nlist_dir}/${nlist}_${probe}_time.txt"
            res_recall_path="${nlist_dir}/${nlist}_${probe}_recall.txt"
            ${executor} -nlist $nlist \
                        -dim $dim \
                        -use_db $use_db \
                        -db $db_path \
                        -learn_file $learn_file \
                        -base_file $base_file \
                        -query_file $query_file \
                        -gt_file $gt_file \
                        -probes $probe \
                        -big_ann_data false \
                        -save_time_file $res_time_path \
                        -save_recall_file $res_recall_path \
                        -use_vec_num true \
                        -num_vecs 1000000
        fi
    done
done
