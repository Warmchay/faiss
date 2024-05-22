import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import matplotlib

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

    dir_path = "/data1/wq/bigann/result/ivf_flat_sift50M/" + str(args.nlists)
    files = os.listdir(dir_path)
    for file in files:
        if not os.path.isdir(file):
            file_elements = file.strip().split('_')
            file_path = dir_path + "/" + file
            if file_elements[-1][0] == 't':
                # time file
                with open(file_path, 'r') as f:
                    for line in f:
                        elements = line.strip().split('\t')
                        nprobes.append(int(elements[0]))
                        time_load.append(float(elements[1]))
                        time_train_train_set.append(float(elements[2]))
                        time_add_base_set.append(float(elements[3]))
                        time_search.append(float(elements[4]))
                        time_computing.append(float(elements[5]))
            else:
                # recall file
                with open(file_path, 'r') as f:
                    for line in f:
                        elements = line.strip().split('\t')
                        r_1.append(float(elements[1]))
                        r_10.append(float(elements[2]))
                        r_100.append(float(elements[3]))
    fig, ax1 = plt.subplots(figsize=(12,6))

    # 绘制柱状图
    index = np.arange(len(nprobes))
    width = 0.15
    ax1.bar(index-2*width, time_load, width=width, color='orange', label='Load data')
    ax1.bar(index-width, time_train_train_set, width=width, color='green', label='Train data')
    ax1.bar(index, time_add_base_set, width=width, color='red', label='Add data')
    ax1.bar(index+width, time_search, width=width, color='purple', label='Search data')
    ax1.bar(index+2*width, time_computing, width=width, color='yellow', label='Compute recall')
    ax1.set_xlabel('Probes')
    ax1.set_ylabel('Time(sec)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 12000)
    ax1.legend(loc='upper left')

    # 绘制折线图
    ax2 = ax1.twinx()
    ax2.plot(index, r_1, marker='o', color='blue', label='Recall@1')
    ax2.plot(index, r_10, marker='o', color='red', label='Recall@10')
    ax2.plot(index, r_100, marker='o', color='green', label='Recall@100')
    ax2.set_ylabel('value', color='k')
    ax2.tick_params(axis='y', labelcolor='k')

    ax2.legend(loc='upper right')
    
    plt.xticks(index, nprobes)
    fig.suptitle(str(args.nlists))
    
    save_fig_path="/home/wq/code/faiss/demos/rocksdb_ivf/fig/" + str(args.nlists) + ".png"
    plt.savefig(save_fig_path, dpi=1000)