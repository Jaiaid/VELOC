#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    Measures how long VELOC_Checkpoint_mem() stalls the application, i.e. the
    time until the application memory has been captured (device D2D/D2H or host
    copy). The memory is split into 3 protected regions holding random data; after
    the timed loop the last checkpoint is restored and verified. Flushing to the filesystem continues in the background; the time
    that takes to drain is reported separately as Checkpoint_wait.

    Usage: test_ckpt_stall <cfg_file> <device|host> [size_MiB=1024] [iterations=5]
    Output (one line per iteration + summary):
      ITER <i> begin=<s> mem=<s> end=<s> wait=<s> stall_MBps=<MB/s>
      SUMMARY mem_stall_avg=<s> mem_stall_min=<s> mem_stall_max=<s> MiB=<n> type=<t>
*/

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <cfg_file> <device|host> [size_MiB] [iterations]\n", argv[0]);
        return 1;
    }
    bool on_device = strcmp(argv[2], "device") == 0;
    size_t mib = argc > 3 ? strtoul(argv[3], NULL, 10) : 1024UL;
    int iters = argc > 4 ? atoi(argv[4]) : 5;
    size_t bytes = mib << 20;

    if (VELOC_Init_single(0, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        return 2;
    }

    std::vector<test_region_t> regs;
    make_regions(regs, bytes, random_base_seed(), on_device ? REGIONS_ALL_DEVICE : REGIONS_ALL_HOST);
    fill_all(regs);
    for (auto &r : regs)
        protect_region(r);

    double sum = 0, mn = 1e30, mx = 0;
    for (int i = 1; i <= iters; i++) {
        // previous checkpoint must be fully flushed so every iteration starts clean
        VELOC_Checkpoint_wait();

        double t0 = now_sec();
        if (VELOC_Checkpoint_begin("test_ckpt_stall", i) != VELOC_SUCCESS) { printf("begin failed\n"); return 6; }
        double t1 = now_sec();
        if (VELOC_Checkpoint_mem() != VELOC_SUCCESS) { printf("mem failed\n"); return 6; }
        double t2 = now_sec();
        if (VELOC_Checkpoint_end(1) != VELOC_SUCCESS) { printf("end failed\n"); return 6; }
        double t3 = now_sec();
        if (VELOC_Checkpoint_wait() != VELOC_SUCCESS) { printf("wait failed\n"); return 6; }
        double t4 = now_sec();

        double stall = t2 - t1;
        printf("ITER %d begin=%.6f mem=%.6f end=%.6f wait=%.6f stall_MBps=%.1f\n",
               i, t1 - t0, stall, t3 - t2, t4 - t3, stall > 0 ? (double)mib / stall : 0.0);
        sum += stall;
        if (stall < mn) mn = stall;
        if (stall > mx) mx = stall;
    }
    printf("SUMMARY mem_stall_avg=%.6f mem_stall_min=%.6f mem_stall_max=%.6f MiB=%zu type=%s avg. stall_MBps=%.6f\n",
           sum / iters, mn, mx, mib, on_device ? "device" : "host", sum > 0 ? (double)mib / (sum/iters): 0.0);

    int rc = restart_and_verify(regs, "test_ckpt_stall", iters);
    VELOC_Finalize(1);
    free_regions(regs);
    return rc;
}
