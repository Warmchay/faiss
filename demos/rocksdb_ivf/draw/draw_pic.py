import numpy as np
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n",
                    "--nlists",
                    type=int,
                    help="nlists' amount")

# 主程序
if __name__ == "__main__":
    args = parser.parse_args()
    nprobes = []
    time_load = []
    time_train_train_set = []
    time_add_base_set = []
    time_search = []
    time_computing = []
    r_1 = []
    r_10 = []
    r_100 = []

    dir_path = "/data1/wq/bigann/result/ivf_flat/sift1b" + str(args.nlist)
    files = os.listdir(dir_path)
    for file in files:
        if not os.path.isdir(file):
            file_elements = file.strip().split('_')
            nprobes.append(file_elements[1])
            file_path = open(dir_path + "/" + file)
            if file_elements[-1][0] == 't':
                # time file
                with open(file_path, 'r') as f:
                    for line in f:
                        elements = line.strip().split('\t')
                        time_load.append(elements[0])
                        time_train_train_set.append(elements[1])
                        time_add_base_set.append(elements[2])
                        time_search.append(elements[3])
                        time_computing.append(elements[4])
            else:
                # recall file
                with open(file_path, 'r') as f:
                    for line in f:
                        elements = line.strip().split('\t')
                        r_1.append(elements[0])
                        r_10.append(elements[1])
                        r_100.append(elements[2])

    fig, ax = plt.subplots()

    # 绘制折线图
    ax.plot(nprobes, r_1, marker='o', color='blue', label='Recall@1')
    ax.plot(nprobes, r_10, marker='o', color='red', label='Recall@10')
    ax.plot(nprobes, r_100, marker='o', color='green', label='Recall@100')

    # 绘制柱状图
    width = 0.15
    ax.bar(nprobes - 2*width, time_load, width=width, color='orange', label='Load data')
    ax.bar(nprobes - width, time_train_train_set, width=width, color='green', label='Train data')
    ax.bar(nprobes, time_add_base_set, width=width, color='red', label='Add data')
    ax.bar(nprobes + width, time_search, width=width, color='purple', label='Search data')
    ax.bar(nprobes + 2*width, time_computing, width=width, color='yellow', label='Compute recall')
    
    save_fig_path="/home/wq/code/faiss/demos/rocksdb_ivf/fig/" + str(args.nlist) + ".png"
    fig.savefig(save_fig_path)