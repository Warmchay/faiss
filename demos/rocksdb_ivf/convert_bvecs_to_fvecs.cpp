#include <gflags/gflags.h>
#include <cstdio>
#include "convert_bvecs_to_fvecs.h"

DEFINE_string(input_file, 
              "", 
              "Input file's loc");

DEFINE_string(output_file, 
              "", 
              "Output file's loc");

std::vector<float> convertToFloatVector(const std::vector<char>& charVector) {
    std::vector<float> floatVector;
    for (const char& c : charVector) {
        floatVector.push_back(static_cast<float>(c));
    }
    return floatVector;
}

int main(int argc, char* argv[]) {
    // std::ifstream inputFile(FLAGS_input_file.c_str());
    // std::ofstream outputFile(FLAGS_output_file.c_str());
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::ifstream inputFile(FLAGS_input_file.c_str());
    if (!inputFile.is_open()) {
        fprintf(stderr, "could not open %s\n", FLAGS_input_file.c_str());
    }

    std::ofstream outputFile(FLAGS_output_file.c_str());
    if (!outputFile.is_open()) {
        fprintf(stderr, "could not open %s\n", FLAGS_output_file.c_str());
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::vector<char> charVector;
        std::stringstream ss(line);
        char c;
        while (ss >> c) {
            charVector.push_back(c);
        }

        std::vector<float> floatVector = convertToFloatVector(charVector);

        for (const float& f : floatVector) {
            outputFile << f;
        }
        outputFile << std::endl;
    }

    printf("Input File:%s\nOutput File:%s\n", FLAGS_input_file.c_str(), FLAGS_output_file.c_str());

    inputFile.close();
    outputFile.close();

    return 0;
}