
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    with open(output_file_path, 'w') as output_file:
        for line in lines:
            parts = line.strip().split(',')
            list_no = int(parts[0].split(':')[1])
            list_size = int(parts[1].split(':')[1])
            output_file.write(f'{list_no}\t{list_size}\n')

# 使用函数
if __name__ == "__main__":
    input_file_path = "/home/wq/code/faiss/demos/rocksdb_ivf/draw/list_2500.txt"
    output_file_path = "/home/wq/code/faiss/demos/rocksdb_ivf/draw/list_2500_done.txt"
    process_file(input_file_path, output_file_path)
