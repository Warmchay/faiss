# Use for process file to left the first element in every line
input_learn_file="/data1/wq/bigann/bigann_learn_without_header.bvecs"
input_base_file="/data1/wq/bigann/bigann_base_without_header.bvecs"
input_query_file="/data1/wq/bigann/bigann_query_without_header.bvecs"

output_learn_file="/data1/wq/bigann/bigann_learn_without_header.fvecs"
output_base_file="/data1/wq/bigann/bigann_base_without_header.fvecs"
output_query_file="/data1/wq/bigann/bigann_query_without_header.fvecs"

executor="/home/wq/code/faiss/demos/rocksdb_ivf/build/convert_bvecs_to_fvecs"

$executor --input_file ${input_learn_file} --output_file ${output_learn_file}
$executor --input_file ${input_base_file}  --output_file ${output_base_file}
$executor --input_file ${input_query_file} --output_file ${output_query_file}