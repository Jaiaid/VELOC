#ifndef GPU_TEST_UTIL_H
#define GPU_TEST_UTIL_H

#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <cuda_runtime.h>

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

#endif
