#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    Allocates application memory on a chosen GPU, protects it with VELOC (which
    lazily creates the GPU staging buffer) and then sleeps so that an external
    observer (test_gpuid.sh, via nvidia-smi) can verify that this process holds
    GPU memory on that GPU only. Also checkpoints, to show the buffer works on
    that GPU.

    Usage: test_gpuid <cfg_file> <gpu_id> [hold_seconds=10] [app_MiB=256]
    Prints "PID <pid> GPU <gpu_id> READY" once the buffer exists.
*/

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <cfg_file> <gpu_id> [hold_seconds] [app_MiB]\n", argv[0]);
        return 1;
    }
    int dev = atoi(argv[2]);
    int hold = argc > 3 ? atoi(argv[3]) : 10;
    size_t bytes = (argc > 4 ? strtoul(argv[4], NULL, 10) : 256UL) << 20;

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (dev >= ngpu) {
        printf("SKIP: gpu %d requested but only %d present\n", dev, ngpu);
        return 77;
    }
    if (VELOC_Init_single(0, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        return 2;
    }

    CUDA_CHECK(cudaSetDevice(dev));
    void *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0x5a, bytes));
    VELOC_Mem_protect(0, d_data, bytes, 1);

    printf("PID %d GPU %d READY\n", (int)getpid(), dev);
    fflush(stdout);
    sleep(hold);

    if (VELOC_Checkpoint("test_gpuid", 1) != VELOC_SUCCESS) {
        printf("Error checkpointing!\n");
        return 6;
    }
    VELOC_Checkpoint_wait();
    VELOC_Finalize(1);
    cudaFree(d_data);
    return 0;
}
