cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
SET(CMAKE_C_COMPILER "/usr/bin/gcc-12")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
project (ROCKSDB_IVF)

set(CMAKE_BUILD_TYPE Release)

find_package(faiss REQUIRED)

find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h PATHS "")
include_directories(${ROCKSDB_INCLUDE_DIR})
find_library(ROCKSDB_LIB rocksdb)
message(STATUS "finding ${ROCKSDB_LIB}")

# find_package(RocksDB REQUIRED)

add_subdirectory(gflags)

add_executable(rocksdb_ivf faiss_rocksdb_ivf.cpp RocksDBInvertedLists.cpp)
add_executable(convert_bvecs_to_fvecs convert_bvecs_to_fvecs.cpp)
add_executable(process_header_file process_header_file.cpp)

target_link_libraries(convert_bvecs_to_fvecs gflags::gflags)
target_link_libraries(process_header_file gflags::gflags)
target_link_libraries(rocksdb_ivf faiss gflags::gflags ${ROCKSDB_LIB} -lpthread -lz -lbz2 -lsnappy -lzstd -ldl)
