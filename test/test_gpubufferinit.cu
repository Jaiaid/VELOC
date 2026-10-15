#include <cstdio>
#include <cstdlib>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    Tests that VELOC creates its GPU staging buffer lazily, on the first
    VELOC_Mem_protect() of a device pointer:
      1. protecting a host region must NOT allocate device memory,
      2. protecting a device region must allocate ~ slots * chunk bytes on the GPU,
      3. protecting a second device region must NOT allocate another buffer.
    Free device memory is sampled with cudaMemGetInfo around each step.

    Usage: test_gpubufferinit <cfg_file> [gpu_id=0] [slots=8] [chunk_MiB=64]
    "slots" must match per-gpu_stage_slots in the cfg (default 8) and "chunk_MiB"
    the compile-time VELOC_XFER_CHUNK (default 64 MiB).
*/

#define MiB (1024UL * 1024UL)

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <cfg_file> [gpu_id] [slots] [chunk_MiB]\n", argv[0]);
        return 1;
    }
    int dev = argc > 2 ? atoi(argv[2]) : 0;
    int slots = argc > 3 ? atoi(argv[3]) : 8;
    size_t chunk = (argc > 4 ? strtoul(argv[4], NULL, 10) : 64UL) * MiB;
    size_t expected = (size_t)slots * chunk;
    const size_t app_bytes = 16 * MiB;
    const size_t tol = 16 * MiB; // allocator granularity / context noise

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (dev >= ngpu) {
        printf("SKIP: gpu %d requested but only %d present\n", dev, ngpu);
        return 77;
    }
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaFree(0)); // create the context up front so it is not counted

    if (VELOC_Init_single(0, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        return 2;
    }

    char *h_data = (char *)malloc(app_bytes);
    void *d_data = NULL, *d_data2 = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, app_bytes));
    CUDA_CHECK(cudaMalloc(&d_data2, app_bytes));
    int fail = 0;

    // 1. host region: no device buffer expected
    size_t f0 = device_free_bytes(dev);
    VELOC_Mem_protect(0, h_data, app_bytes, 1);
    size_t f1 = device_free_bytes(dev);
    long host_delta = (long)f0 - (long)f1;
    printf("host protect:   device memory consumed = %ld MiB\n", host_delta / (long)MiB);
    if (host_delta > (long)tol) {
        printf("FAIL: protecting host memory allocated a GPU buffer\n");
        fail = 1;
    }

    // 2. first device region: buffer expected
    VELOC_Mem_protect(1, d_data, app_bytes, 1);
    size_t f2 = device_free_bytes(dev);
    long dev_delta = (long)f1 - (long)f2;
    printf("device protect: device memory consumed = %ld MiB (expected ~%zu MiB)\n",
           dev_delta / (long)MiB, expected / MiB);
    if (dev_delta < (long)(expected - tol)) {
        printf("FAIL: GPU buffer was not created on GPU %d\n", dev);
        fail = 1;
    }

    // 3. second device region: buffer must be reused
    VELOC_Mem_protect(2, d_data2, app_bytes, 1);
    size_t f3 = device_free_bytes(dev);
    long again_delta = (long)f2 - (long)f3;
    printf("2nd device protect: device memory consumed = %ld MiB\n", again_delta / (long)MiB);
    if (again_delta > (long)tol) {
        printf("FAIL: GPU buffer was created more than once\n");
        fail = 1;
    }

    VELOC_Finalize(1);
    cudaFree(d_data);
    cudaFree(d_data2);
    free(h_data);
    printf("test_gpubufferinit: %s\n", fail ? "FAILED" : "PASSED");
    return fail;
}
