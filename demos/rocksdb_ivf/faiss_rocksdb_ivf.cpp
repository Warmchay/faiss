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
              "Data: Use learn dataset to construct initial groups");

DEFINE_bool(big_ann_data, 
            false, 
            "Data: Use bvecs dataset");

DEFINE_string(base_file, 
              "", 
              "Dataset: Use base dataset to enlarge data scopes");

DEFINE_string(query_file, 
              "", 
              "Dataset: Use query dataset to test QPS");

DEFINE_string(gt_file, 
              "",
              "Dataset: Use groundtruth dataset to test recall");

DEFINE_string(save_time_file, 
              "", 
              "Dataset: Store time statistics");

DEFINE_string(save_recall_file, 
              "", 
              "Dataset: Store recall statistics");

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
        float* xt;
        if (FLAGS_big_ann_data) {
            unsigned char* xt = bvecs_read(FLAGS_learn_file.c_str(), &dimt, &nt);
        } else {
            float* xt = fvecs_read(FLAGS_learn_file.c_str(), &dimt, &nt);    
        }
        assert(dim == dimt || !"Training set has wrong dim");
        double time_load_train_set = elapsed() -t0;
        printf("[%.3f s] Training on %ld vectors\n", time_load_train_set, nt);
        index.train(nt, xt);
        delete [] xt;

        double time_train_train_set = elapsed() - t0;
        printf("[%.3f s] Loading base set\n", time_train_train_set);
        size_t nb, dimb;
        float* xb;
        if (FLAGS_big_ann_data) {
            unsigned char* xb = bvecs_read(FLAGS_base_file.c_str(), &dimb, &nb);
        } else {
            float* xb = fvecs_read(FLAGS_base_file.c_str(), &dimb, &nb);
        }
        assert(dim == dimb || !"Base set has wrong dim");
        double time_load_base_set = elapsed() - t0;
        printf("[%.3f s] Indexing on %ld vectors\n", time_load_base_set, nb);
        index.add(nb, xb);
        delete [] xb;
        
        double time_add_base_set = elapsed() - t0;
        printf("[%.3f s] Loading queries\n", time_add_base_set);
        size_t nq, dimq;
        float* xq;
        if (FLAGS_big_ann_data) {
            unsigned char* xq = bvecs_read(FLAGS_query_file.c_str(), &dimq, &nq);
        } else {
            float* xq = fvecs_read(FLAGS_query_file.c_str(), &dimq, &nq);
        }
        assert(dim == dimq || !"Query set has wrong dim");

        double time_load_query = elapsed() - t0;
        printf("[%.3f s] Loading ground truth for %ld queries\n", time_load_query, nq);
        size_t ng, kg;
        faiss::idx_t* gt;
        int* xg = ivecs_read(FLAGS_gt_file.c_str(), &kg, &ng);
        assert(ng == nq || !"Ground truth has incorrect nums");
        double time_load_gt = elapsed() - t0;
        gt = new faiss::idx_t[kg * nq];
        for (int i = 0; i < kg; ++i) {
            gt[i] = xg[i];
        }
        delete [] xg;

        faiss::idx_t* I = new faiss::idx_t[nq * kg];
        float* D = new float[nq * kg];
        printf("[%.3f s] Perform a search on %ld queries\n", elapsed() - t0, nq);
        index.search(nq, xq, kg, D, I);

        double time_search = elapsed() - t0;
        printf("[%.3f s] Compute recalls\n", time_search);
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
        double time_compute = elapsed() - t0;
        double r_1 = n_1 / double(nq);
        double r_10 = n_10 / double(nq);
        double r_100 = n_100 / double(nq);

        FILE* fp_time = fopen(FLAGS_save_time_file.c_str(), "a");
        if (!fp_time) {
            fprintf(stderr, "could not open %s\n", FLAGS_save_time_file.c_str());
            perror("");
            abort();
        }
        double time_load = time_load_train_set + time_load_base_set + time_load_query + time_load_gt;
        fprintf(fp_time, 
        "%.4lf\t%.4lf\t%.4lf\t%4.lf\t%.4lf\n", 
                time_load, time_train_train_set, time_add_base_set, time_search, time_compute);
        fclose(fp_time);

        FILE* fp_recall = fopen(FLAGS_save_recall_file.c_str(), "a");
        fprintf(fp_recall, 
                "%.4lf\t%4.lf\t%.4lf\n", 
                r_1, r_10, r_100);
        fclose(fp_recall);

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