#!/usr/bin/bash

# Test for ivf flat in rocksdb
# Define dataset's vars
dim=128
nlists=(100 500 1000 5000 10000 50000 100000)

# Define use rocksdb or not
use_db=true
db_path="/data1/wq/bigann/db"

# Select used files
num_base_vecs=1000000
num_learn_vecs=500000
learn_file="/data1/wq/bigann/bigann_learn_${num_learn_vecs}.fvecs"
base_file="/data1/wq/bigann/bigann_base_${num_base_vecs}.fvecs"
query_file="/data1/wq/bigann/bigann_query.fvecs"
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
                        -save_recall_file $res_recall_path
                        # -use_vec_num false \
                        # -num_vecs $num_vecs_amount
        fi
    done
done
