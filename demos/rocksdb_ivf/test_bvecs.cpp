#include "test_bvecs.h"
#include <gflags/gflags.h>
#include <cstdint>

DEFINE_string(input_file, 
              "", 
              "Input file's loc");

DEFINE_string(output_file, 
              "", 
              "Output file's loc");

DEFINE_int32(num_vecs, 
             100000, 
             "Translate vec's amount");

DEFINE_bool(use_vec, 
            false,
            "Decide to use amount of vecs");

void convert_bvecs_to_fvecs(const std::string& input_filename, const std::string& output_filename) {
    std::ifstream input_file(input_filename, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error: Unable to open input file " << input_filename << std::endl;
        return;
    }
    std::ofstream output_file(output_filename, std::ios::binary);
    if (!output_file.is_open()) {
        std::cerr << "Error: Unable to create output file " << output_filename << std::endl;
        input_file.close();
        return;
    }

    int dim;
    char value;
    if (FLAGS_use_vec) {
        int64_t i = 0;
        while (input_file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
            if (i == FLAGS_num_vecs) {
                break;
            }
            std::vector<float> fvec(dim);
            for (int i = 0; i < dim; ++i) {
                input_file.read(reinterpret_cast<char*>(&value), sizeof(char));
                fvec[i] = static_cast<float>(value);
            }
            ++i;
            // Write the dimension of the fvec
            output_file.write(reinterpret_cast<char*>(&dim), sizeof(int));
            // Write the fvec data
            output_file.write(reinterpret_cast<char*>(fvec.data()), dim * sizeof(float));
        }
    } else {
        while (input_file.read(reinterpret_cast<char*>(&dim), sizeof(int))) {
            std::vector<float> fvec(dim);
            for (int i = 0; i < dim; ++i) {
                input_file.read(reinterpret_cast<char*>(&value), sizeof(char));
                fvec[i] = static_cast<float>(value);
            }

            // Write the dimension of the fvec
            output_file.write(reinterpret_cast<char*>(&dim), sizeof(int));
            // Write the fvec data
            output_file.write(reinterpret_cast<char*>(fvec.data()), dim * sizeof(float));
        }
    }


    input_file.close();
    output_file.close();
}

int main(int argc, char* argv[]) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    convert_bvecs_to_fvecs(FLAGS_input_file.c_str(), FLAGS_output_file.c_str());
    return 0;
}