from faiss.contrib import datasets
from faiss.contrib.exhaustive_search import knn_ground_truth
from faiss.contrib import vecs_io

ds = datasets.DatasetFloatBigANN("/data1/wq/bigann", 100)

print("computing GT for ", ds)

D, I = knn_ground_truth(
  ds.get_queries(),
  ds.database_iterator(bs=65535),
  k=100
)

vecs_io.ivecs_write("/data1/wq/bigann/gnd/self_gt_100M.ivecs", I)