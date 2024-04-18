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

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/random.h>

#include <gflags/gflags.h>

using namespace faiss;

/* Define Optinal Vars **/
DEFINE_int32(nlist, 
             100,
             "Data: Nums of centroids");

DEFINE_int64(dim, 
             128, 
             "Data: Vector data's dimesion");

DEFINE_bool(use_pq, 
            false, 
            "Use PQ: To compress vec");

DEFINE_int64(M, 
             4, 
             "Use PQ: Split origin vec to M subvecs");

DEFINE_int64(nbits_per_idx, 
             4,
             "Use PQ: Codebook's index bit nums");

DEFINE_bool(use_db, 
              true, 
              "Use Disk: To use disk");

DEFINE_string(db, 
              "", 
              "Use Disk: Decide where to store data");

DEFINE_string(learn_file, 
              "",
              "Data: use learn dataset to construct initial groups");

DEFINE_string(base_file, 
              "", 
              "Dataset: use base dataset to enlarge data scopes");

DEFINE_string(query_file, 
              "", 
              "Dataset: use query dataset to test QPS");

DEFINE_string(gt_file, 
              "",
              "Dataset: Use groundtruth dataset to test recall");
DEFINE_int32(probes, 
             10,
             "Data: Decide detecting probes");

// main
int main(int argc, char* argv[]) {
    try {
        double t0 = elapsed();
        ::gflags::ParseCommandLineFlags(&argc, &argv, true);

        size_t dim = FLAGS_dim;
        size_t nlist = FLAGS_nlist;
        IndexFlatL2 quantizer(dim);
        
        IndexIVFFlat index(&quantizer, dim, nlist);            

        if (FLAGS_use_pq) {
            IndexIVFPQ index(&quantizer, dim, nlist, FLAGS_M, FLAGS_nbits_per_idx);
        }

        if (FLAGS_use_db) {
            faiss_rocksdb::RocksDBInvertedLists ril(
                FLAGS_db.c_str(), nlist, index.code_size
            );
            index.replace_invlists(&ril, false);
        }

        printf("[%.3f s] Loading training set\n", elapsed() - t0);
        size_t nt, dimt;
        float* xt = fvecs_read(FLAGS_learn_file.c_str(), &dimt, &nt);
        assert(dim == dimt || !"Training set has wrong dim");
        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
        index.train(nt, xt);
        delete [] xt;

        printf("[%.3f s] Loading base set\n", elapsed() - t0);
        size_t nb, dimb;
        float* xb = fvecs_read(FLAGS_base_file.c_str(), &dimb, &nb);
        assert(dim == dimb || !"Base set has wrong dim");
        printf("[%.3f s] Indexing on %ld vectors\n", elapsed() - t0, nb);
        index.add(nb, xb);
        delete [] xb;
        
        printf("[%.3f s] Loading queries\n", elapsed() - t0);
        size_t nq, dimq;
        float* xq = fvecs_read(FLAGS_query_file.c_str(), &dimq, &nq);
        assert(dim == dimq || !"Query set has wrong dim");

        printf("[%.3f s] Loading ground truth for %ld queries\n", elapsed() - t0, nq);
        size_t ng, kg;
        faiss::idx_t* gt;
        int* xg = ivecs_read(FLAGS_gt_file.c_str(), &kg, &ng);
        assert(ng == nq || !"Ground truth has incorrect nums");
        gt = new faiss::idx_t[kg * nq];
        for (int i = 0; i < kg; ++i) {
            gt[i] = xg[i];
        }
        delete [] xg;

        faiss::idx_t* I = new faiss::idx_t[nq * kg];
        float* D = new float[nq * kg];
        printf("[%.3f s] Perform a search on %ld queries\n", elapsed() - t0, nq);
        index.search(nq, xq, kg, D, I);

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for (int i = 0; i < nq; i++) {
            int gt_nn = gt[i * kg];
            for (int j = 0; j < kg; j++) {
                if (I[i * kg + j] == gt_nn) {
                    if (j < 1)
                        n_1++;
                    if (j < 10)
                        n_10++;
                    if (j < 100)
                        n_100++;
                }
            }
        }
        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));
        delete[] I;
        delete[] D;
        delete[] xq;
        delete[] gt;

    } catch (FaissException& e) {
        std::cerr << e.what() << '\n';
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    } catch (...) {
        std::cerr << "Unrecognized exception!\n";
    }
    return 0;
}