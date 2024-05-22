import numpy as np
import matplotlib.pyplot as plt


# 主程序
if __name__ == "__main__":
    nlists = []
    probes = []

    file_path = "/data1/wq/bigann/result/ivf_flat_sift100M/recall95_time_cost_24_0521.txt"

    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')
            nlists.append(int(elements[0]))
            probes.append(int(elements[3]))

    fig, ax1 = plt.subplots(figsize=(12,6))

    # 绘制柱状图
    index = np.arange(len(nlists))
    width = 0.3

    bar1 = ax1.bar(index, probes, width=width, color='#426666', label='probes')
    line1 = ax1.plot(index, probes, marker='x', color='#d3b17d')
    plt.bar_label(bar1, labels=probes, padding=0.2)

    ax1.set_xlabel('nlists')
    ax1.set_ylabel('probes')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 75)
    ax1.legend(loc='upper right')
    
    plt.xticks(index, nlists)
    
    save_fig_path="/home/wq/code/faiss/demos/rocksdb_ivf/fig/" + "100M_nlist_probes.png"
    plt.savefig(save_fig_path, dpi=500)