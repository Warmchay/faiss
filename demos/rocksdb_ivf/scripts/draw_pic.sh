#!/usr/bin/bash

# Use for drawing picture
nlists=(100 500 1000 5000 10000)

execute_program="/home/wq/code/faiss/demos/rocksdb_ivf/draw/draw_pic.py"
python_path="/home/wq/.conda/envs/rocksdb_ivf/bin/python"

for nlist in "${nlists[@]}"
do
    ${python_path} ${execute_program} --nlist ${nlist}
done