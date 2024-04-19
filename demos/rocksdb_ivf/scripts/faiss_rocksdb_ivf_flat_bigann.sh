# Test for ivf flat in rocksdb

# Define dataset's vars
nlists=(100 500 1000 5000 10000 50000 100000)
dim=128

# Define use rocksdb or not
use_db=true
db_path="/data1/wq/bigann/db"

# Select used files
learn_file="/data1/wq/bigann/bigann_learn.bvecs"
base_file="/data1/wq/bigann/bigann_base.bvecs"
query_file="/data1/wq/bigann/bigann_query.bvecs"
gt_file="/data1/wq/bigann/bigann_gtn.ivecs"

# Define searching vars
probes=(10 20 40 80 160 320 640 1280 2560 5120 10240)

# Decide res files location
res_dir="/data1/wq/bigann/result/ivf_flat/sift1b"


# Define executor
executor=""

for nlist in $nlists
    for probe in $probes
        if [ $nlist -gt $probe]
        then
            res_time_path="${res_dir}/${nlist}_${probe}_time.txt"
            res_recall_path = "${res_dir}/${nlist}_${probe}_recall.txt"
            ${executor} -nlist $nlist \
                        -dim $dim \
                        -use_db $use_db \
                        -db $db_path \
                        -learn_file $learn_file \
                        -base_file $base_file \
                        -query_file $query_file \
                        -gt_file $gt_file \
                        -probes $probe \
                        -big_ann_data true \
                        -save_time_file $res_time_path \
                        -save_recall_file $res_recall_path
        fi
    end
end
