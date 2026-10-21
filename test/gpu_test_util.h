#ifndef GPU_TEST_UTIL_H
#define GPU_TEST_UTIL_H

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "include/veloc.h"

#define CUDA_CHECK(call)                                                        \
    do {                                                                       \
        cudaError_t _e = (call);                                              \
        if (_e != cudaSuccess) {                                              \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(_e));                                   \
            exit(3);                                                          \
        }                                                                    \
    } while (0)

static inline double now_sec() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// free device memory (bytes) of the given device, without leaving it as the current device
static inline size_t device_free_bytes(int dev) {
    int prev = 0;
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaGetDevice(&prev));
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    CUDA_CHECK(cudaSetDevice(prev));
    return free_b;
}


// ---------------------------------------------------------------------------
// Multi-region helpers: several protected regions (device and host) filled with
// a reproducible pseudo-random pattern (splitmix64 stream keyed by a per-region
// seed), so that restart verification does not need to keep a reference copy.
// ---------------------------------------------------------------------------

struct test_region_t {
    int id;
    void *ptr;
    size_t bytes;
    bool device;
    uint64_t seed;
};

enum region_mix_t {
    REGIONS_MIXED,       // 3 device regions (1:2:3 split of total) + one 8 MiB host region
    REGIONS_ALL_DEVICE,  // 3 device regions
    REGIONS_ALL_HOST     // 3 host regions
};

struct pattern_rng_t {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

static const size_t PATTERN_CHUNK = 16UL << 20; // multiple of 8, keeps the stream continuous

static inline void pattern_bytes(pattern_rng_t &r, unsigned char *p, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t v = r.next();
        memcpy(p + i, &v, 8);
    }
    if (i < n) {
        uint64_t v = r.next();
        memcpy(p + i, &v, n - i);
    }
}

// fresh random base seed (different on every run); printed so failures can be reproduced
static inline uint64_t random_base_seed() {
    std::random_device rd;
    return ((uint64_t)rd() << 32) ^ rd();
}

// allocates (not yet protected) regions of about total_bytes of "bulk" memory
static inline void make_regions(std::vector<test_region_t> &regs, size_t total_bytes,
                                uint64_t base_seed, region_mix_t mix) {
    size_t unit = total_bytes / 6;
    size_t sizes[3] = {unit, 2 * unit, total_bytes - 3 * unit};
    for (int i = 0; i < 3; i++) {
        test_region_t r = {i, NULL, sizes[i], mix != REGIONS_ALL_HOST, base_seed + 1 + i};
        if (r.device)
            CUDA_CHECK(cudaMalloc(&r.ptr, r.bytes));
        else
            r.ptr = malloc(r.bytes);
        regs.push_back(r);
    }
    if (mix == REGIONS_MIXED) {
        test_region_t h = {3, malloc(8UL << 20), 8UL << 20, false, base_seed + 100};
        regs.push_back(h);
    }
}

static inline void protect_region(const test_region_t &r) {
    if (VELOC_Mem_protect(r.id, r.ptr, r.bytes, 1) != VELOC_SUCCESS) {
        printf("VELOC_Mem_protect failed for region %d\n", r.id);
        exit(3);
    }
}

static inline void free_regions(std::vector<test_region_t> &regs) {
    for (auto &r : regs) {
        if (r.device) cudaFree(r.ptr); else free(r.ptr);
    }
    regs.clear();
}

static inline void fill_region(const test_region_t &r) {
    std::vector<unsigned char> buf(PATTERN_CHUNK);
    pattern_rng_t rng = {r.seed};
    for (size_t off = 0; off < r.bytes; off += PATTERN_CHUNK) {
        size_t n = r.bytes - off < PATTERN_CHUNK ? r.bytes - off : PATTERN_CHUNK;
        pattern_bytes(rng, buf.data(), n);
        if (r.device)
            CUDA_CHECK(cudaMemcpy((char *)r.ptr + off, buf.data(), n, cudaMemcpyHostToDevice));
        else
            memcpy((char *)r.ptr + off, buf.data(), n);
    }
}

static inline void clobber_region(const test_region_t &r) {
    if (r.device)
        CUDA_CHECK(cudaMemset(r.ptr, 0, r.bytes));
    else
        memset(r.ptr, 0, r.bytes);
}

// returns true if region content equals the pattern generated from its seed
static inline bool verify_region(const test_region_t &r) {
    std::vector<unsigned char> expect(PATTERN_CHUNK), actual(PATTERN_CHUNK);
    pattern_rng_t rng = {r.seed};
    for (size_t off = 0; off < r.bytes; off += PATTERN_CHUNK) {
        size_t n = r.bytes - off < PATTERN_CHUNK ? r.bytes - off : PATTERN_CHUNK;
        pattern_bytes(rng, expect.data(), n);
        const unsigned char *got = (const unsigned char *)r.ptr + off;
        if (r.device) {
            CUDA_CHECK(cudaMemcpy(actual.data(), got, n, cudaMemcpyDeviceToHost));
            got = actual.data();
        }
        if (memcmp(expect.data(), got, n) != 0) {
            size_t i = 0;
            while (expect[i] == got[i]) i++;
            printf("MISMATCH in region %d (%s) at byte %zu: got 0x%02x expected 0x%02x\n",
                   r.id, r.device ? "device" : "host", off + i, got[i], expect[i]);
            return false;
        }
    }
    return true;
}

static inline void fill_all(std::vector<test_region_t> &regs) { for (auto &r : regs) fill_region(r); }
static inline void clobber_all(std::vector<test_region_t> &regs) { for (auto &r : regs) clobber_region(r); }
static inline bool verify_all(std::vector<test_region_t> &regs) {
    bool ok = true;
    for (auto &r : regs) ok &= verify_region(r);
    return ok;
}

// Restart from checkpoint `name`/`version` over clobbered regions and verify
// that every region got its random pattern back. Returns 0 on success.
static inline int restart_and_verify(std::vector<test_region_t> &regs, const char *name, int version) {
    clobber_all(regs);
    if (VELOC_Restart(name, version) != VELOC_SUCCESS) {
        printf("Error restarting!\n");
        return 4;
    }
    if (!verify_all(regs)) {
        printf("FAIL: restored data differs from the checkpointed random pattern\n");
        return 5;
    }
    printf("restart verification OK (%zu regions)\n", regs.size());
    return 0;
}

// checkpoint (data must already be filled), wait for flush, then restart+verify
static inline int checkpoint_restart_verify(std::vector<test_region_t> &regs, const char *name, int version) {
    if (VELOC_Checkpoint(name, version) != VELOC_SUCCESS) {
        printf("Error checkpointing!\n");
        return 6;
    }
    VELOC_Checkpoint_wait();
    return restart_and_verify(regs, name, version);
}

#endif
