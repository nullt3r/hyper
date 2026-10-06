// Minimal read-only GGUF v3 reader: mmaps the file, exposes metadata and tensor views.
#pragma once
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace hyper {

enum class GType : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18, IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21,
    IQ2_S = 22, IQ4_XS = 23, I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, IQ1_M = 29, BF16 = 30,
};

const char * gtype_name(GType t);
// bytes per block and elements per block for the types hyper understands
size_t gtype_block_bytes(GType t);
size_t gtype_block_elems(GType t);

struct GTensor {
    std::string name;
    GType type;
    std::vector<int64_t> ne;   // ggml order: ne[0] is the contiguous (row) dimension
    const uint8_t * data = nullptr;
    size_t nbytes = 0;

    int64_t nelements() const { int64_t n = 1; for (auto x : ne) n *= x; return n; }
    int64_t rows() const { return nelements() / ne[0]; }
    size_t row_bytes() const { return (size_t) (ne[0] / gtype_block_elems(type)) * gtype_block_bytes(type); }
};

using GValue = std::variant<int64_t, double, bool, std::string, std::vector<int64_t>, std::vector<double>, std::vector<std::string>>;

class GGUF {
public:
    explicit GGUF(const std::string & path);
    ~GGUF();
    GGUF(const GGUF &) = delete;
    GGUF & operator=(const GGUF &) = delete;

    bool has(const std::string & key) const { return kv_.count(key) > 0; }
    int64_t get_int(const std::string & key) const;
    int64_t get_int(const std::string & key, int64_t def) const { return has(key) ? get_int(key) : def; }
    double get_float(const std::string & key) const;
    double get_float(const std::string & key, double def) const { return has(key) ? get_float(key) : def; }
    std::string get_str(const std::string & key) const;
    std::vector<int64_t> get_int_arr(const std::string & key) const;

    const GTensor * tensor(const std::string & name) const;   // nullptr if missing
    const GTensor & need(const std::string & name) const;     // throws if missing
    const std::map<std::string, GTensor> & tensors() const { return tensors_; }
    const std::string & arch() const { return arch_; }

private:
    int fd_ = -1;
    uint8_t * map_ = nullptr;
    size_t size_ = 0;
    std::string arch_;
    std::map<std::string, GValue> kv_;
    std::map<std::string, GTensor> tensors_;
};

} // namespace hyper
