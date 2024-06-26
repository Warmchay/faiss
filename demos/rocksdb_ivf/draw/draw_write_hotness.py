import matplotlib.pyplot as plt

# from pylab import mpl
# mpl.rcParams["font.sans-serif"] = ["SimHei"]

# import matplotlib as mpl
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


# 主程序
if __name__ == "__main__":
    listNo = []
    write_hotness = []

    file_path = "/home/wq/code/faiss/demos/rocksdb_ivf/draw/list_2500_done.txt"

    with open(file_path, 'r') as f:
        for line in f:
            elements = line.strip().split('\t')
            listNo.append(int(elements[0]))
            write_hotness.append(int(elements[1]))

    fig, ax1 = plt.subplots(figsize=(12,6))

    # 绘制柱状图
    width = 0.7

    # bar1 = ax1.bar(listNo, write_hotness, width=width,color='#483D8B', label='向量数')
    bar1 = ax1.bar(listNo, write_hotness, width=width,color='#DA70D6', label='向量数')
    # plt.bar_label(bar1, labels=probes, padding=0.2)
    ax1.set_ylim(0, 20000)
    ax1.set_xlabel('分区编号')
    ax1.set_ylabel('向量数量')
    ax1.legend(loc='upper right')
        
    save_fig_path="/home/wq/code/faiss/demos/rocksdb_ivf/fig/" + "list_size_2500.png"
    plt.savefig(save_fig_path, dpi=500)