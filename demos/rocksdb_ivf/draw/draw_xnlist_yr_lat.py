import numpy as np
import matplotlib.pyplot as plt


# 主程序
if __name__ == "__main__":
    nlists = []
    recalls = []
    lats = []

    file_path = "/data1/wq/bigann/result/ivf_flat_sift100M/recall95_time_cost_24_0521.txt"

    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')
            nlists.append(int(elements[0]))
            recalls.append(float(elements[1]) * 100)
            lats.append(int(elements[2]))

    fig, ax1 = plt.subplots(figsize=(12,6))

    # 绘制柱状图
    index = np.arange(len(nlists))
    width = 0.3

    bar1 = ax1.bar(index, lats, width=width, color='#9ed048', label='Latency(s)')
    line = ax1.plot(index, lats, marker='x', color='#758a99')
    plt.bar_label(bar1, labels=lats, padding=0.2)

    ax1.set_xlabel('nlist')
    ax1.set_ylabel('Latency(sec)')
    ax1.tick_params(axis='y')
    ax1.set_ylim(0, 5000)


    # 绘制折线图
    ax2 = ax1.twinx()
    line1, = ax2.plot(index, recalls, marker='o', color='#d9b611', label='Recall@1(%)')

    ax2.set_ylabel('Recall@1(%)', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim(90, 102)

    lns = [bar1, line1]
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc="upper right")

    plt.xticks(index, nlists)
    
    save_fig_path="/home/wq/code/faiss/demos/rocksdb_ivf/fig/" + "100M_nlist_recall_lat.png"
    plt.savefig(save_fig_path, dpi=500)