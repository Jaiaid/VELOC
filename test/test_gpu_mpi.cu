#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    MPI version of heatdis_gpu (see build/test/test_gpu.sh). Each rank owns
    several protected regions (3 device regions on GPU rank % num_gpus plus a
    host region) filled with random data, and a host iteration counter. Run it
    twice with the same config and the same seed: the first run checkpoints, the
    second run finds the checkpoint, restarts over clobbered regions and verifies
    that every region on every rank got its random data back.

    Usage: mpirun -np N test_gpu_mpi <cfg_file> [array_MiB=256] [seed]
    The seed (base of the per-rank, per-region patterns) must be identical in
    both runs; test_gpu_mpi.sh generates a random one.
*/

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <cfg_file> [array_MiB] [seed]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    size_t bytes = (argc > 2 ? strtoul(argv[2], NULL, 10) : 256UL) << 20;
    uint64_t seed = argc > 3 ? strtoull(argv[3], NULL, 10) : 12345;
    seed += (uint64_t)rank * 1000003ULL; // different data on every rank

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(rank % ngpu));

    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int iter = 0, rc = 0;
    std::vector<test_region_t> regs;
    make_regions(regs, bytes, seed, REGIONS_MIXED);
    VELOC_Mem_protect(100, &iter, 1, sizeof(int));
    for (auto &r : regs)
        protect_region(r);

    int v = VELOC_Restart_test("test_gpu_mpi", 0);
    if (v > 0) {
        if (rank == 0) printf("Previous checkpoint found (version %d), restarting...\n", v);
        rc = restart_and_verify(regs, "test_gpu_mpi", v);
        if (rc == 0 && iter != 3) {
            printf("RANK %d: FAIL iteration counter is %d, expected 3\n", rank, iter);
            rc = 5;
        }
        printf("RANK %d (GPU %d): restart verification %s (iter=%d)\n", rank, rank % ngpu, rc ? "FAIL" : "OK", iter);
    } else {
        iter = 3;
        fill_all(regs);
        if (VELOC_Checkpoint("test_gpu_mpi", iter) != VELOC_SUCCESS) {
            printf("RANK %d: error checkpointing!\n", rank);
            rc = 6;
        } else {
            printf("RANK %d (GPU %d): checkpoint OK (iter=%d)\n", rank, rank % ngpu, iter);
        }
        VELOC_Checkpoint_wait();
    }

    int total_rc = 0;
    MPI_Allreduce(&rc, &total_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    VELOC_Finalize(v > 0 ? 1 : 0);
    free_regions(regs);
    MPI_Finalize();
    return total_rc;
}
