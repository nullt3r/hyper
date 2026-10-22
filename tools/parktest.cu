// ConvPark: park / resume round trips through the pooled pinned chunks restore every span byte for byte
// (HYPER_PARK_CHUNK_KB small: rows cross chunk boundaries); also times parking a large set
#include "park.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace hyper;
struct Snap { int pos = 0; std::vector<float *> h; };

int main(int argc, char ** argv) {
    const int n_max = argc > 1 ? atoi(argv[1]) : 3000;
    cudaStream_t st;
    cudaSetDevice(0);
    if (cudaStreamCreate(&st) != cudaSuccess) { printf("stream failed\n"); return 1; }
    // spans like engine4's: K / V (2 heads, pitch for n_max rows), pooled keys (div 4), an odd row size
    struct B { size_t row; int heads, div; void * d; size_t pitch; };
    std::vector<B> bs = {{512, 2, 1, nullptr, 0}, {512, 2, 1, nullptr, 0}, {136, 1, 4, nullptr, 0}, {72, 3, 1, nullptr, 0}};
    ConvPark<Snap>::Dev dev;
    dev.id = 0;
    dev.stream = st;
    for (auto & b : bs) {
        b.pitch = ((size_t) n_max + 1) * b.row;
        if (cudaMalloc(&b.d, b.pitch * b.heads) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
        dev.spans.push_back({b.d, b.row, b.pitch, b.heads, b.div});
    }
    auto fill = [&](unsigned seed) {   // the device contents of a conversation
        for (auto & b : bs) {
            std::vector<unsigned char> h(b.pitch * b.heads);
            for (size_t i = 0; i < h.size(); ++i) h[i] = (unsigned char) ((i * 2654435761u + seed * 97u) >> 13);
            cudaMemcpy(b.d, h.data(), h.size(), cudaMemcpyHostToDevice);
        }
    };
    auto snapshot = [&] {
        std::vector<std::vector<unsigned char>> v;
        for (auto & b : bs) { v.emplace_back(b.pitch * b.heads); cudaMemcpy(v.back().data(), b.d, v.back().size(), cudaMemcpyDeviceToHost); }
        return v;
    };
    auto same_prefix = [&](const std::vector<std::vector<unsigned char>> & a, int n) {   // positions [0, n) of every head
        auto cur = snapshot();
        for (size_t k = 0; k < bs.size(); ++k) {
            const size_t w = (bs[k].div == 1 ? (size_t) n : (size_t) (n / bs[k].div + 1)) * bs[k].row;
            for (int h = 0; h < bs[k].heads; ++h)
                if (memcmp(a[k].data() + h * bs[k].pitch, cur[k].data() + h * bs[k].pitch, w)) {
                    size_t i = 0;
                    while (a[k][h * bs[k].pitch + i] == cur[k][h * bs[k].pitch + i]) ++i;
                    printf("  span %zu head %d differs at byte %zu of %zu (row %zu)\n", k, h, i, w, i / bs[k].row);
                    return false;
                }
        }
        return true;
    };
    ConvPark<Snap> park;
    park.min_park = 100;
    park.min_gain = 50;
    std::vector<Snap> snaps;
    std::vector<std::vector<float *>> pool;
    std::vector<ConvPark<Snap>::Dev> devs = {dev};
    int fails = 0;
    // conversation A grows (n_a), side requests S (unrelated) park it, then A comes back
    std::vector<int> A, hist;
    for (int round = 0; round < 12; ++round) {
        const int n_a = n_max / 3 + round * (n_max / 20);
        A.clear();
        for (int i = 0; i < n_a; ++i) A.push_back(1000 + i);
        // A is current, with its contents
        hist = A;
        snaps.clear();
        snaps.push_back({n_a - 10, {}});
        fill(round + 1);
        const auto a_img = snapshot();
        // side request S: parks A
        std::vector<int> S;
        // (even rounds: a new side conversation -> A is parked; odd: the previous one again -> swapped back, A parked in its place)
        for (int i = 0; i < 200 + round; ++i) S.push_back(5 + 100000 * (round / 2) + i);
        const int L = park.resolve(S, hist, snaps, pool, devs, "parktest");
        hist = S;
        snaps.clear();
        snaps.push_back({150, {}});
        fill(100 + round);   // S overwrites the device
        // A extended comes back
        std::vector<int> A2 = A;
        A2.push_back(7);
        const int L2 = park.resolve(A2, hist, snaps, pool, devs, "parktest");
        const bool ok = L2 == n_a && same_prefix(a_img, n_a);
        fails += !ok;
        printf("round %d: A %d tokens, side L %d, back L %d: %s\n", round, n_a, L, L2, ok ? "identical" : "DIFFERENT");
    }
    printf("PARKTEST %s\n", fails ? "FAILED" : "OK");
    return fails != 0;
}
