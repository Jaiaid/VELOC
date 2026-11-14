#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    Once VELOC_Checkpoint_mem() returns, the application may overwrite its
    device memory: the snapshot must already be taken.

    A long spin kernel is first queued on the legacy default stream. The
    engine's capture copy (cudaMemcpy D2D on the legacy default stream, from the
    engine thread) is queued behind it, so the snapshot cannot be complete before
    the spin kernel has finished.

    Check 1 (deterministic): the spin kernel must have finished (an event
      recorded right after it has completed) when Checkpoint_mem() returns. The
      legacy stream itself may still be busy with the engine's own later work
      (demotion to host). This assumes the capture copy is ordered on the legacy
      stream; if the engine moves to its own stream, this check has to follow it.
    Check 2: the application zeroes its regions on a non-blocking stream right
      after the return and the restart must still give back the original data.
      In practice the driver delays that memset while the engine thread sits in
      its synchronous D2H demotion, so this rarely fails on its own.

    Usage: test_gpu_snapshot <cfg_file> [size_MiB=256] [spin_ms=300]
*/

__global__ void spin_kernel(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles)
        ;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <cfg_file> [size_MiB] [spin_ms]\n", argv[0]);
        return 1;
    }
    int ngpu = 0;
    if (cudaGetDeviceCount(&ngpu) != cudaSuccess || ngpu == 0) {
        printf("no GPU available, skipping\n");
        return 77;
    }
    size_t mib = argc > 2 ? strtoul(argv[2], NULL, 10) : 256UL;
    int spin_ms = argc > 3 ? atoi(argv[3]) : 300;

    if (VELOC_Init_single(0, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        return 2;
    }

    std::vector<test_region_t> regs;
    make_regions(regs, mib << 20, random_base_seed(), REGIONS_ALL_DEVICE);
    // Keep only a few regions so that every chunk fits a free GPU staging slot
    // (default 8): a chunk that has to wait for a slot, or falls back to D2H,
    // blocks behind the spin kernel and would hide the premature return.
    const size_t keep = 4;
    for (size_t i = keep; i < regs.size(); i++)
        CUDA_CHECK(cudaFree(regs[i].ptr));
    regs.resize(keep);
    fill_all(regs);
    for (auto &r : regs)
        protect_region(r);
    CUDA_CHECK(cudaDeviceSynchronize());

    // warm-up checkpoint: one-time setup (e.g. pinned allocations) synchronizes
    // the device and would hide the premature return on the first checkpoint
    if (VELOC_Checkpoint("test_gpu_snapshot", 0) != VELOC_SUCCESS || VELOC_Checkpoint_wait() != VELOC_SUCCESS) {
        printf("warm-up checkpoint failed\n");
        return 6;
    }

    int clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    cudaStream_t app_stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&app_stream, cudaStreamNonBlocking));

    // unrelated, still-running application work on the legacy default stream
    cudaEvent_t spin_done;
    CUDA_CHECK(cudaEventCreateWithFlags(&spin_done, cudaEventDisableTiming));
    spin_kernel<<<1, 1>>>((long long)clock_khz * spin_ms);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(spin_done, 0));

    double t0 = now_sec();
    if (VELOC_Checkpoint_begin("test_gpu_snapshot", 1) != VELOC_SUCCESS ||
        VELOC_Checkpoint_mem() != VELOC_SUCCESS) {
        printf("checkpoint_mem failed\n");
        return 6;
    }
    double stall = now_sec() - t0;

    cudaError_t spin = cudaEventQuery(spin_done); // work queued before the snapshot copy
    // the application resumes and overwrites its data
    double t1 = now_sec();
    for (auto &r : regs)
        CUDA_CHECK(cudaMemsetAsync(r.ptr, 0, r.bytes, app_stream));
    CUDA_CHECK(cudaStreamSynchronize(app_stream));
    double t2 = now_sec();

    printf("Checkpoint_mem returned after %.3f s (spin kernel %d ms), spin kernel %s, overwrite took %.3f s\n",
           stall, spin_ms, spin == cudaSuccess ? "finished" : "still running", t2 - t1);
    if (VELOC_Checkpoint_end(1) != VELOC_SUCCESS || VELOC_Checkpoint_wait() != VELOC_SUCCESS) {
        printf("checkpoint_end/wait failed\n");
        return 6;
    }

    // restore the original pattern: the checkpoint must hold it, not the zeroes
    int rc = restart_and_verify(regs, "test_gpu_snapshot", 1);
    if (rc != 0)
        printf("FAIL: the checkpoint captured data written after Checkpoint_mem() returned\n");
    if (spin != cudaSuccess) {
        printf("FAIL: Checkpoint_mem() returned while the snapshot copy was still queued behind earlier GPU work\n");
        rc = 7;
    }
    VELOC_Finalize(1);
    CUDA_CHECK(cudaStreamDestroy(app_stream));
    CUDA_CHECK(cudaEventDestroy(spin_done));
    free_regions(regs);
    return rc;
}
