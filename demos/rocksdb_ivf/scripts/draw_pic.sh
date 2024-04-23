#!/usr/bin/bash

# Use for drawing picture
nlists=(100 500 1000 5000 10000 50000 100000)

execute_program="/home/wq/code/faiss/demos/rocksdb_ivf/draw/draw_pic.py"

for nlist in ${nlists}
do
    python ${execute_program} --nlist ${nlist}
done