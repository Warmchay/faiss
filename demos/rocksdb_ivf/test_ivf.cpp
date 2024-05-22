#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>
#include <fstream>
#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>

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

DEFINE_uint64(nlist, 100, "");
// DEFINE_string(save_time_file, "", "");
// DEFINE_string(save_recall_file, "", "");
DEFINE_int64(probes, 10, "");
// DEFINE_string(db, "", "");

int main(int argc, char* argv[]) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    double t0 = elapsed();
    size_t dim = 128, nlist = FLAGS_nlist;
    IndexFlatL2 quantizer(dim);
    IndexIVFFlat index(&quantizer, dim, nlist);
    double time_load = 0, time_train_train_set = 0, time_add_base_set = 0, time_search = 0, time_compute = 0;
    double time_pre_load = 0, time_pre_other = 0;
    std::string db_path_str = "/data1/wq/bigann/db/" + std::to_string(FLAGS_nlist) + "_" + std::to_string(FLAGS_probes);
    std::string fp_time_path_str = "/data1/wq/bigann/result/ivf_flat_sift100M/" + std::to_string(nlist) + "_time.txt";
    std::string fp_recall_path_str = "/data1/wq/bigann/result/ivf_flat_sift100M/" + std::to_string(nlist) + "_recall.txt";
    // const char* fp_time_path = "/data1/wq/bigann/result/ivf_flat_sift50M/100_time.txt";
    // const char* fp_recall_path = "/data1/wq/bigann/result/ivf_flat_sift50M/100_recall.txt";
    const char* db_path = db_path_str.c_str();

    if (mkdir(db_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
        std::cout << "Create dir fails" << std::endl;
    } else {
        std::cout << "Create dir: " << db_path << std::endl;
    }

    const char* fp_time_path = fp_time_path_str.c_str();
    const char* fp_recall_path = fp_recall_path_str.c_str();

    faiss_rocksdb::RocksDBInvertedLists ril(
        db_path, nlist, index.code_size
    );
    index.replace_invlists(&ril, false);

    {
        size_t dimt, nt;

        printf("[%.3f s] Loading training set\n", elapsed() - t0);
        time_pre_load = elapsed();
        float* xt = fvecs_read("/data1/wq/bigann/bigann_learn.fvecs", &dimt, &nt);
        assert(dimt == dim);
        time_load += elapsed() - time_pre_load;

        printf("[%.3f s] Training on %ld vectors\n", elapsed() - t0, nt);
        time_pre_other = elapsed();
        index.train(nt, xt);
        time_train_train_set = elapsed() - time_pre_other;
        delete[] xt;
    }

    {
        size_t nb, dimb;

        printf("[%.3f s] Loading base set\n", elapsed() - t0);
        time_pre_load = elapsed();
        float* xb = fvecs_read("/data1/wq/bigann/bigann_base_100M.fvecs", &dimb, &nb);
        assert(dimb == dim);
        time_load += elapsed() - time_pre_load;

        printf("[%.3f s] Indexing on %ld vectors\n", elapsed() - t0, nb);
        time_pre_other = elapsed();
        index.add(nb, xb);
        time_add_base_set = elapsed() - time_pre_other;
        delete[] xb; 
    }

    size_t nq, dimq;
    float* xq;
    
    {
        printf("[%.3f s] Loading queries\n", elapsed() - t0);
        time_pre_load = elapsed();
        xq = fvecs_read("/data1/wq/bigann/bigann_query.fvecs", &dimq, &nq);
        assert(dimq == dim);
        time_load += elapsed() - time_pre_load;
    }

    size_t ng, dimg;
    faiss::idx_t* gt;

    {    
        printf("[%.3f s] Loading ground truth for %ld queries\n", elapsed() - t0, nq);
        time_pre_load = elapsed();
        int* xg = ivecs_read("/data1/wq/bigann/gnd/self_gt_idx_100M.ivecs", &dimg, &ng);
        assert(nq == ng);
        time_load += elapsed() - time_pre_load;

        gt = new idx_t[dimg * ng];
        for (int i = 0; i < dimg * ng; ++i) {
            gt[i] = xg[i];
        }
        delete[] xg; 
    }

    {
        idx_t* I = new idx_t[nq * dimg];
        float* D = new float[nq * dimg];
        printf("[%.3f s] Perform a search on %ld queries\n", elapsed() - t0, nq);
        // std::vector<int64_t> nprobes = {10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240};
        // for (auto& probes : nprobes) {
        int64_t probes = FLAGS_probes;
        index.nprobe = probes;
        time_pre_other = elapsed();
        index.search(nq, xq, dimg, D, I);
        time_search = elapsed() - time_pre_other;

        printf("[%.3f s] Compute recalls\n", elapsed() - t0);
        time_pre_other = elapsed();
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
        double time_compute = elapsed() - time_pre_other;
        double r_1 = n_1 / double(nq);
        double r_10 = n_10 / double(nq);
        double r_100 = n_100 / double(nq);

        FILE* fp_time = fopen(fp_time_path, "a");
        FILE* fp_recall = fopen(fp_recall_path, "a");
        // std::ofstream fp_time, fp_recall;
        // fp_time.open(fp_time_path, std::ios::app);
        // fp_recall.open(fp_recall_path, std::ios::app);
        // std::cout.precision(4);
        // fp_time << probes << "\t" << time_load << "\t" << time_train_train_set << "\t" 
        //         << time_add_base_set << "\t" << time_search << "\t" << time_compute << std::endl;
        // fp_time.close();

        // fp_recall << r_1 << "\t" << r_10 << "\t" << r_100 << std::endl;
        // fp_recall.close();
        // if (!(fp_time && fp_recall)) {
        //     fprintf(stderr, "could not open files\n");
        //     perror("");
        //     abort();
        // }


        fprintf(fp_time, 
        "%ld\t%.4lf\t%.4lf\t%.4lf\t%4.lf\t%.4lf\n", 
                probes, time_load, time_train_train_set, time_add_base_set, time_search, time_compute);
        fclose(fp_time);

        fprintf(fp_recall, 
                "%ld\t%.4lf\t%.4lf\t%.4lf\n", 
                probes, r_1, r_10, r_100);
        fclose(fp_recall);

        printf("R@1 = %.4f\n", r_1);
        printf("R@10 = %.4f\n", r_10);
        printf("R@100 = %.4f\n", r_100);
        delete[] I;
        delete[] D;
        delete[] xq;
        delete[] gt;
        // }   
    }
}