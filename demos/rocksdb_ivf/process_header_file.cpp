#include "process_header_file.h"

#include <gflags/gflags.h>
#include <cstdio>

DEFINE_string(input_file, 
              "", 
              "Input file's loc");

DEFINE_string(output_file, 
              "", 
              "Output file's loc");

void process_file_header(const char* input_file_name, const char* output_file_name) {
    std::ifstream input_file(input_file_name);
    if (!input_file.is_open()) {
        fprintf(stderr, "could not open %s\n", input_file_name);
    }

    std::ofstream output_file(output_file_name);
    if (!output_file.is_open()) {
        fprintf(stderr, "could not open %s\n", output_file_name);
    }

    std::string line;
    while (std::getline(input_file, line)) {
        std::istringstream iss(line);
        std::string firstElement;
        iss >> firstElement;
        std::string restOfLine;
        std::getline(iss, restOfLine);

        output_file << restOfLine << std::endl;
    }

    input_file.close();
    output_file.close();
}

int main(int argc, char* argv[]) {
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    process_file_header(FLAGS_input_file.c_str(), FLAGS_output_file.c_str());

    printf("Input file: %s\nOutput file:%s", FLAGS_input_file.c_str(), FLAGS_output_file.c_str());
    
    return 0;
}



