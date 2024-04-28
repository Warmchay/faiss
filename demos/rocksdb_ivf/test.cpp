#include <cassert>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>

#include "RocksDBInvertedLists.h"
#include "faiss/IndexIVF.h"
#include "faiss/IndexIVFPQ.h"
#include "faiss/MetricType.h"

#include <bits/types/FILE.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/random.h>

#include <gflags/gflags.h>

using namespace faiss;

DEFINE_int64(nlist, 100, "");
DEFINE_string(save_time_file, "", "");
DEFINE_string(save_recall_file, "", "");
DEFINE_int64(probes, 10, "");

int main() {
    double t0 = elapsed();
    size_t dim = 128, nlist = FLAGS_nlist;
    IndexFlatL2 quantizer(dim);
    IndexIVFFlat index(&quantizer, dim, nlist);
    double time_load, time_train_train_set, time_add_base_set, time_search, time_compute;
    faiss_rocksdb::RocksDBInvertedLists ril(
        "/data1/wq/bigann/db", nlist, index.code_size
    );
    index.replace_invlists(&ril, false);

    {
        double time_load_train = elapsed() - t0;
        printf("[%.3f s] Loading training set\n", elapsed() - t0);
        size_t dimt, nt;
        float* xt = fvecs_read("/data1/wq/bigann/bigann_learn.fvecs", &dimt, &nt);
        assert(dimt == dim);
        printf("[%.3f s] Training on %ld vectors\n", elapsed()-t0, nt);
        index.train(nt, xt);
        time_train_train_set = elapsed() - time_load_train;
        time_load = time_load_train;
        delete[] xt;
    }

    {
        double time_load_base = elapsed() - t0;
        printf("[%.3f s] Loading base set\n", elapsed() - t0);
        size_t nb, dimb;
        float* xb = fvecs_read("/data1/wq/bigann/bigann_base_50M.fvecs", &dimb, &nb);
        assert(dimb == dim);
        printf("[%.3f s] Indexing on %ld vectors\n", elapsed() - t0, nb);
        index.add(nb, xb);
        time_add_base_set = elapsed() - time_load_base;
        time_load += time_load_base;
        delete[] xb; 
    }

    size_t nq, dimq;
    float* xq;
    
    {
        double time_load_query = 0;
        printf("[%.3f s] Loading queries\n", elapsed() - t0);
        xq = fvecs_read("/data1/wq/bigann/bigann_query.fvecs", &dimq, &nq);
        assert(dimq == dim);
        time_load += time_load_query;
    }

    size_t ng, dimg;
    faiss::idx_t* gt;

    {
        printf("[%.3f s] Loading ground truth for %ld queries\n", elapsed() - t0, nq);
        int* xg = ivecs_read("/data1/wq/bigann/gnd/idx_1M.ivecs", &dimg, &ng);
        
        gt = new idx_t[dimg * ng];
        assert(nq == ng);
        for (int i = 0; i < dimg * ng; ++i) {
            gt[i] = xg[i];
        }
        delete[] xg; 
    }

    {
        idx_t* I = new idx_t[nq * dimg];
        float* D = new float[nq * dimg];
        printf("[%.3f s] Perform a search on %ld queries\n", elapsed() - t0, nq);
        index.search(nq, xq, dimg, D, I);
        printf("[%.3f s] Compute recalls\n", elapsed() - t0);
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * dimg];
            for (int j = 0; j < dimg; j++) {
                if (I[i * dimg + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        double time_compute = elapsed() - t0;
        double r_1 = n_1 / double(nq);
        double r_10 = n_10 / double(nq);
        double r_100 = n_100 / double(nq);

        printf("R@1 = %.4f\n", r_1);
        printf("R@10 = %.4f\n", r_10);
        printf("R@100 = %.4f\n", r_100);
        delete[] I;
        delete[] D;
        delete[] xq;
        delete[] gt;
    }

}