#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>

namespace hyper {

const char * gtype_name(GType t) {
    switch (t) {
        case GType::F32:  return "F32";
        case GType::F16:  return "F16";
        case GType::BF16: return "BF16";
        case GType::Q8_0: return "Q8_0";
        case GType::Q4_0: return "Q4_0";
        case GType::Q4_1: return "Q4_1";
        case GType::Q5_0: return "Q5_0";
        case GType::Q5_1: return "Q5_1";
        case GType::IQ4_NL: return "IQ4_NL";
        case GType::Q4_K: return "Q4_K";
        case GType::Q5_K: return "Q5_K";
        case GType::Q6_K: return "Q6_K";
        default:          return "?";
    }
}

size_t gtype_block_bytes(GType t) {
    switch (t) {
        case GType::F32:  return 4;
        case GType::F16:  return 2;
        case GType::BF16: return 2;
        case GType::Q8_0: return 34;
        case GType::Q4_0: return 18;
        case GType::Q4_1: return 20;
        case GType::Q5_0: return 22;
        case GType::Q5_1: return 24;
        case GType::IQ4_NL: return 18;
        case GType::Q4_K: return 144;
        case GType::Q5_K: return 176;
        case GType::Q6_K: return 210;
        default: throw std::runtime_error(std::string("unsupported tensor type ") + std::to_string((uint32_t) t));
    }
}

size_t gtype_block_elems(GType t) {
    switch (t) {
        case GType::F32: case GType::F16: case GType::BF16: return 1;
        case GType::Q8_0: case GType::Q4_0: case GType::Q4_1: case GType::Q5_0: case GType::Q5_1: case GType::IQ4_NL: return 32;
        case GType::Q4_K: case GType::Q5_K: case GType::Q6_K: return 256;
        default: throw std::runtime_error(std::string("unsupported tensor type ") + std::to_string((uint32_t) t));
    }
}

namespace {

enum : uint32_t { T_U8 = 0, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 };

struct Cursor {
    const uint8_t * p;
    const uint8_t * end;
    template <typename T> T get() {
        if (p + sizeof(T) > end) throw std::runtime_error("gguf: truncated header");
        T v; memcpy(&v, p, sizeof(T)); p += sizeof(T); return v;
    }
    std::string str() {
        uint64_t n = get<uint64_t>();
        if (p + n > end) throw std::runtime_error("gguf: truncated string");
        std::string s((const char *) p, n); p += n; return s;
    }
};

bool is_int_type(uint32_t t) { return t <= T_I32 || t == T_U64 || t == T_I64; }

int64_t read_int(Cursor & c, uint32_t t) {
    switch (t) {
        case T_U8: return c.get<uint8_t>();   case T_I8:  return c.get<int8_t>();
        case T_U16: return c.get<uint16_t>(); case T_I16: return c.get<int16_t>();
        case T_U32: return c.get<uint32_t>(); case T_I32: return c.get<int32_t>();
        case T_U64: return (int64_t) c.get<uint64_t>(); case T_I64: return c.get<int64_t>();
        case T_BOOL: return c.get<uint8_t>();
    }
    throw std::runtime_error("gguf: not an int");
}

GValue read_value(Cursor & c, uint32_t t) {
    if (is_int_type(t)) return read_int(c, t);
    switch (t) {
        case T_F32: return (double) c.get<float>();
        case T_F64: return c.get<double>();
        case T_BOOL: return (bool) c.get<uint8_t>();
        case T_STR: return c.str();
        case T_ARR: {
            uint32_t at = c.get<uint32_t>();
            uint64_t n = c.get<uint64_t>();
            if (at == T_STR) { std::vector<std::string> v; v.reserve(n); for (uint64_t i = 0; i < n; ++i) v.push_back(c.str()); return v; }
            if (at == T_F32 || at == T_F64) { std::vector<double> v; v.reserve(n); for (uint64_t i = 0; i < n; ++i) v.push_back(at == T_F32 ? c.get<float>() : c.get<double>()); return v; }
            std::vector<int64_t> v; v.reserve(n); for (uint64_t i = 0; i < n; ++i) v.push_back(read_int(c, at)); return v;
        }
    }
    throw std::runtime_error("gguf: unknown value type " + std::to_string(t));
}

} // namespace

GGUF::GGUF(const std::string & path) {
    // split model: name-00001-of-0000N.gguf
    const size_t at = path.rfind("-00001-of-");
    if (at != std::string::npos && path.size() > at + 15) {
        const int n = std::stoi(path.substr(at + 10, 5));
        for (int i = 1; i <= n; ++i) {
            char num[8];
            snprintf(num, sizeof num, "%05d", i);
            std::string p = path;
            p.replace(at + 1, 5, num);
            load_shard(p, i == 1);
        }
    } else {
        load_shard(path, true);
    }
    arch_ = get_str("general.architecture");
}

void GGUF::load_shard(const std::string & path, bool first) {
    Map m;
    m.fd = open(path.c_str(), O_RDONLY);
    if (m.fd < 0) throw std::runtime_error("gguf: cannot open " + path);
    struct stat st; fstat(m.fd, &st); m.size = st.st_size;
    m.p = (uint8_t *) mmap(nullptr, m.size, PROT_READ, MAP_SHARED, m.fd, 0);
    if (m.p == MAP_FAILED) throw std::runtime_error("gguf: mmap failed");
    maps_.push_back(m);

    Cursor c{m.p, m.p + m.size};
    if (c.get<uint32_t>() != 0x46554747) throw std::runtime_error("gguf: bad magic");
    uint32_t version = c.get<uint32_t>();
    if (version < 2) throw std::runtime_error("gguf: unsupported version");
    uint64_t n_tensors = c.get<uint64_t>();
    uint64_t n_kv = c.get<uint64_t>();
    std::map<std::string, GValue> kv;
    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        uint32_t t = c.get<uint32_t>();
        kv[key] = read_value(c, t);
    }
    size_t alignment = 32;
    if (auto it = kv.find("general.alignment"); it != kv.end()) alignment = (size_t) std::get<int64_t>(it->second);
    if (first) kv_ = kv;

    std::vector<std::pair<std::string, uint64_t>> offsets;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        GTensor t;
        t.name = c.str();
        uint32_t nd = c.get<uint32_t>();
        for (uint32_t d = 0; d < nd; ++d) t.ne.push_back((int64_t) c.get<uint64_t>());
        t.type = (GType) c.get<uint32_t>();
        uint64_t off = c.get<uint64_t>();
        tensors_[t.name] = t;
        offsets.emplace_back(t.name, off);
    }
    size_t data_start = ((size_t) (c.p - m.p) + alignment - 1) / alignment * alignment;
    for (auto & [name, off] : offsets) {
        GTensor & t = tensors_[name];
        t.data = m.p + data_start + off;
        t.nbytes = t.row_bytes() * (size_t) t.rows();
        if (t.data + t.nbytes > m.p + m.size) throw std::runtime_error("gguf: tensor out of bounds: " + name);
    }
}

GGUF::~GGUF() {
    for (auto & m : maps_) {
        if (m.p && m.p != MAP_FAILED) munmap(m.p, m.size);
        if (m.fd >= 0) close(m.fd);
    }
}

int64_t GGUF::get_int(const std::string & key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    if (auto p = std::get_if<int64_t>(&it->second)) return *p;
    if (auto p = std::get_if<bool>(&it->second)) return *p;
    throw std::runtime_error("gguf: key is not an int: " + key);
}

double GGUF::get_float(const std::string & key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    if (auto p = std::get_if<double>(&it->second)) return *p;
    if (auto p = std::get_if<int64_t>(&it->second)) return (double) *p;
    throw std::runtime_error("gguf: key is not a float: " + key);
}

std::string GGUF::get_str(const std::string & key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    if (auto p = std::get_if<std::string>(&it->second)) return *p;
    throw std::runtime_error("gguf: key is not a string: " + key);
}

std::vector<int64_t> GGUF::get_int_arr(const std::string & key) const {
    auto it = kv_.find(key);
    if (it == kv_.end()) throw std::runtime_error("gguf: missing key " + key);
    if (auto p = std::get_if<std::vector<int64_t>>(&it->second)) return *p;
    if (auto p = std::get_if<int64_t>(&it->second)) return {*p};
    throw std::runtime_error("gguf: key is not an int array: " + key);
}

const GTensor * GGUF::tensor(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const GTensor & GGUF::need(const std::string & name) const {
    auto t = tensor(name);
    if (!t) throw std::runtime_error("gguf: missing tensor " + name);
    return *t;
}

} // namespace hyper
