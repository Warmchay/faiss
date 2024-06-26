// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "RocksDBInvertedLists.h"

#include <bits/types/FILE.h>
#include <sys/stat.h>
#include <sys/time.h>



#include <faiss/impl/FaissAssert.h>
#include <cstdio>
#include <string>

using namespace faiss;

namespace faiss_rocksdb {

RocksDBInvertedListsIterator::RocksDBInvertedListsIterator(
        rocksdb::DB* db,
        size_t list_no,
        size_t code_size)
        : InvertedListsIterator(),
          it(db->NewIterator(rocksdb::ReadOptions())),
          list_no(list_no),
          code_size(code_size),
          codes(code_size) {
    it->Seek(rocksdb::Slice(
            reinterpret_cast<const char*>(&list_no), sizeof(size_t)));
}

bool RocksDBInvertedListsIterator::is_available() const {
    return it->Valid() &&
            it->key().starts_with(rocksdb::Slice(
                    reinterpret_cast<const char*>(&list_no), sizeof(size_t)));
}

void RocksDBInvertedListsIterator::next() {
    it->Next();
}

std::pair<idx_t, const uint8_t*> RocksDBInvertedListsIterator::
        get_id_and_codes() {
    idx_t id =
            *reinterpret_cast<const idx_t*>(&it->key().data()[sizeof(size_t)]);
    assert(code_size == it->value().size());
    return {id, reinterpret_cast<const uint8_t*>(it->value().data())};
}

RocksDBInvertedLists::RocksDBInvertedLists(
        const char* db_directory,
        size_t nlist,
        size_t code_size)
        : InvertedLists(nlist, code_size) {
    use_iterator = true;

    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::DB* db;
    rocksdb::Status status = rocksdb::DB::Open(options, db_directory, &db);
    db_ = std::unique_ptr<rocksdb::DB>(db);
    assert(status.ok());
}

size_t RocksDBInvertedLists::list_size(size_t /*list_no*/) const {
    FAISS_THROW_MSG("list_size is not supported");
}

const uint8_t* RocksDBInvertedLists::get_codes(size_t /*list_no*/) const {
    FAISS_THROW_MSG("get_codes is not supported");
}

const idx_t* RocksDBInvertedLists::get_ids(size_t /*list_no*/) const {
    FAISS_THROW_MSG("get_ids is not supported");
}

size_t RocksDBInvertedLists::add_entries(
        size_t list_no,
        size_t n_entry,
        const idx_t* ids,
        const uint8_t* code) {
    rocksdb::WriteOptions wo;
    std::vector<char> key(sizeof(size_t) + sizeof(idx_t));
    memcpy(key.data(), &list_no, sizeof(size_t));
    write_hist_[list_no] += 1;
    for (size_t i = 0; i < n_entry; i++) {
        memcpy(key.data() + sizeof(size_t), ids + i, sizeof(idx_t));
        rocksdb::Status status = db_->Put(
                wo,
                rocksdb::Slice(key.data(), key.size()),
                rocksdb::Slice(
                        reinterpret_cast<const char*>(code + i * code_size),
                        code_size));
        assert(status.ok());
    }
    return 0; // ignored
}

void RocksDBInvertedLists::generate_write_hotness_hist() {
    std::string save_write_hotness_name = "/data1/wq/bigann/result/ivf_flat_sift100M/write_hotness.txt";
    FILE* fp_write_hotness = fopen(save_write_hotness_name.c_str(), "a");
    for (auto write_group : write_hist_) {
        fprintf(fp_write_hotness, 
                "%ld\t%ld\n",
                write_group.first, write_group.second);
    }
}

void RocksDBInvertedLists::update_entries(
        size_t /*list_no*/,
        size_t /*offset*/,
        size_t /*n_entry*/,
        const idx_t* /*ids*/,
        const uint8_t* /*code*/) {
    FAISS_THROW_MSG("update_entries is not supported");
}

void RocksDBInvertedLists::resize(size_t /*list_no*/, size_t /*new_size*/) {
    FAISS_THROW_MSG("resize is not supported");
}

InvertedListsIterator* RocksDBInvertedLists::get_iterator(
        size_t list_no,
        void* inverted_list_context) const {
    return new RocksDBInvertedListsIterator(db_.get(), list_no, code_size);
}

} // namespace faiss_rocksdb

unsigned char* bvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    struct stat st;
    size_t d = 128;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % d == 0 || !"weird file size");
    size_t n = sz / d;

    *d_out = d;
    *n_out = n;
    unsigned char* x = new unsigned char[n * d];
    size_t nr = fread(x, sizeof(unsigned char), n * d, f);
    assert(nr == n * d || !"could not read whole file");
    fclose(f);
    return (unsigned char*)x;
}

unsigned char* bvecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t num_vecs) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    struct stat st;
    size_t d = 128;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % d == 0 || !"weird file size");
    size_t n = num_vecs;
    *d_out = d;
    *n_out = n;
    unsigned char* x = new unsigned char[n * d];
    size_t nr = fread(x, sizeof(unsigned char), n * d, f);
    assert(nr == n * d || !"could not read whole file");
    fclose(f);
    return (unsigned char*)x;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d;
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for (size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t num_vecs) {
    FILE* f = fopen(fname, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d, n;
    // fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    // fvecs: d * 4 + 4
    d = 128;
    n = num_vecs;
    assert(sz % (d * 4) == 0 || !"weird file size");
    *d_out = d;
    *n_out = n;
    float* x = new float[n * d];
    size_t nr = fread(x, sizeof(unsigned char), n * d, f);
    assert(nr == n * d || !"could not read whole file");

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return (int*)fvecs_read(fname, d_out, n_out);
}

int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out, size_t num_vecs) {
    return (int*)fvecs_read(fname, d_out, n_out, num_vecs);
}

double elapsed() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}