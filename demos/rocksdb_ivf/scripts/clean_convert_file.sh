num_base_vecs=50000000
num_learn_vecs=500000

output_learn_file="/data1/wq/bigann/bigann_base_${num_base_vecs}.fvecs"
# output_base_file="/data1/wq/bigann/bigann_learn_${num_learn_vecs}.fvecs"
# output_query_file="/data1/wq/bigann/bigann_query.fvecs"


rm -rf $output_learn_file $output_base_file $output_query_file 