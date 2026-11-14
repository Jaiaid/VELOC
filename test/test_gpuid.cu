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
    that GPU. The application memory is split in several regions (device + a small
    host one) filled with random data; after the hold, the checkpoint is restored
    over clobbered regions and compared with the pattern.

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
    std::vector<test_region_t> regs;
    make_regions(regs, bytes, random_base_seed(), REGIONS_MIXED);
    fill_all(regs);
    for (auto &r : regs)
        protect_region(r);

    printf("PID %d GPU %d READY\n", (int)getpid(), dev);
    fflush(stdout);
    sleep(hold);

    int rc = checkpoint_restart_verify(regs, "test_gpuid", 1);
    VELOC_Finalize(1);
    free_regions(regs);
    return rc;
}
