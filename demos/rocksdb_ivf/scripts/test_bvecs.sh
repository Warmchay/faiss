# Use for testing how to read bvecs files
executor="/home/wq/code/faiss/demos/rocksdb_ivf/build/test_bvecs"
# Base
num_base_vecs=50000000
input_base_file="/data1/wq/bigann/bigann_base.bvecs"
output_base_file="/data1/wq/bigann/bigann_base_50M.fvecs"
${executor} --input_file ${input_base_file} \
            --output_file ${output_base_file} \
            --use_vec true \
            --num_vecs ${num_base_vecs}
# Learn
# num_learn_vecs=500000
input_learn_file="/data1/wq/bigann/bigann_learn.bvecs"
output_learn_file="/data1/wq/bigann/bigann_learn.fvecs"
${executor} --input_file ${input_learn_file} \
            --output_file ${output_learn_file} \
            # --use_vec true \
            # --num_vecs ${num_learn_vecs}

# Query
# input_query_file="/data1/wq/bigann/bigann_query.bvecs"
# output_query_file="/data1/wq/bigann/bigann_query.fvecs"
# ${executor} --input_file ${input_query_file} \
#             --output_file ${output_query_file}