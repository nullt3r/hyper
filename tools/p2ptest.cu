// P2P check: for every device pair, can-access-peer, a kernel on device a writing into / reading from device b's memory
// through peer access, and the copy bandwidth (cudaMemcpyPeer) against the same copy through pinned host memory.
// usage: p2ptest [MiB=256]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA %s (line %d)\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__global__ void k_fill(unsigned * p, size_t n, unsigned seed) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = seed ^ (unsigned) i;
}
__global__ void k_check(const unsigned * p, size_t n, unsigned seed, unsigned * bad) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && p[i] != (seed ^ (unsigned) i)) atomicAdd(bad, 1u);
}

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? atoi(argv[1]) : 256, bytes = mib << 20, n = bytes / 4;
    int nd = 0; CK(cudaGetDeviceCount(&nd));
    std::vector<unsigned *> buf(nd), bad(nd);
    std::vector<void *> host(nd);
    for (int g = 0; g < nd; ++g) { CK(cudaSetDevice(g)); CK(cudaMalloc(&buf[g], bytes)); CK(cudaMalloc(&bad[g], 4)); CK(cudaHostAlloc(&host[g], bytes, cudaHostAllocPortable)); }
    printf("can access peer (row -> column):\n");
    for (int a = 0; a < nd; ++a) { for (int b = 0; b < nd; ++b) { int ok = a == b; if (a != b) CK(cudaDeviceCanAccessPeer(&ok, a, b)); printf(" %d", ok); } printf("\n"); }
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    for (int a = 0; a < nd; ++a)
        for (int b = 0; b < nd; ++b) {
            if (a == b) continue;
            int ok = 0; CK(cudaDeviceCanAccessPeer(&ok, a, b));
            double gbs_peer = 0, gbs_host = 0; unsigned nbad = 0; const char * kern = "-";
            if (ok) {
                CK(cudaSetDevice(a));
                cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { printf("%d -> %d: enable peer access failed: %s\n", a, b, cudaGetErrorString(e)); continue; }
                // a kernel on a writes b's buffer directly; b verifies
                k_fill<<<(unsigned) ((n + 255) / 256), 256>>>(buf[b], n, 0x1234u + a * 16 + b);
                e = cudaDeviceSynchronize();
                if (e != cudaSuccess) { printf("%d -> %d: peer write kernel failed: %s\n", a, b, cudaGetErrorString(e)); kern = "FAIL"; }
                else {
                    CK(cudaSetDevice(b)); CK(cudaMemset(bad[b], 0, 4));
                    k_check<<<(unsigned) ((n + 255) / 256), 256>>>(buf[b], n, 0x1234u + a * 16 + b, bad[b]);
                    CK(cudaMemcpy(&nbad, bad[b], 4, cudaMemcpyDeviceToHost));
                    kern = nbad ? "MISMATCH" : "ok";
                    // timed: the kernel writing over the link
                    CK(cudaSetDevice(a));
                    auto t0 = now();
                    for (int r = 0; r < 4; ++r) k_fill<<<(unsigned) ((n + 255) / 256), 256>>>(buf[b], n, r);
                    CK(cudaDeviceSynchronize());
                    const double kw = 4.0 * bytes / secs(t0, now()) / 1e9;
                    t0 = now();
                    for (int r = 0; r < 4; ++r) CK(cudaMemcpyPeer(buf[b], b, buf[a], a, bytes));
                    CK(cudaDeviceSynchronize());
                    gbs_peer = 4.0 * bytes / secs(t0, now()) / 1e9;
                    printf("%d -> %d: peer kernel write %s (%.1f GB/s), memcpy peer %.1f GB/s", a, b, kern, kw, gbs_peer);
                }
            } else printf("%d -> %d: no peer access", a, b);
            // through the host: a -> pinned -> b
            CK(cudaSetDevice(a));
            auto t0 = now();
            for (int r = 0; r < 4; ++r) {
                CK(cudaSetDevice(a)); CK(cudaMemcpy(host[a], buf[a], bytes, cudaMemcpyDeviceToHost));
                CK(cudaSetDevice(b)); CK(cudaMemcpy(buf[b], host[a], bytes, cudaMemcpyHostToDevice));
            }
            gbs_host = 4.0 * bytes / secs(t0, now()) / 1e9;
            printf(", via host %.1f GB/s\n", gbs_host);
        }
    return 0;
}
