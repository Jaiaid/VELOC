#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <mpi.h>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    MPI test: rank r allocates its application data on GPU (r % num_gpus),
    protects it with VELOC (so VELOC's GPU buffer must follow to that same GPU),
    checkpoints, and holds so test_multigpu_alloc.sh can verify with nvidia-smi
    that every rank's process is resident on its own GPU.

    Usage: mpirun -np N test_multigpu_alloc <cfg_file> [hold_seconds=10] [app_MiB=256]
    Prints "RANK <r> PID <pid> GPU <g> READY" per rank.
*/

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (argc < 2) {
        if (rank == 0)
            printf("Usage: %s <cfg_file> [hold_seconds] [app_MiB]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    int hold = argc > 2 ? atoi(argv[2]) : 10;
    size_t bytes = (argc > 3 ? strtoul(argv[3], NULL, 10) : 256UL) << 20;

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu < 2 && rank == 0)
        printf("WARNING: only %d GPU(s), ranks will share GPUs\n", ngpu);
    int dev = rank % ngpu;

    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    CUDA_CHECK(cudaSetDevice(dev));
    // several regions per rank (device + small host), random per-rank data
    std::vector<test_region_t> regs;
    make_regions(regs, bytes, random_base_seed(), REGIONS_MIXED);
    fill_all(regs);
    for (auto &r : regs)
        protect_region(r);

    printf("RANK %d PID %d GPU %d READY\n", rank, (int)getpid(), dev);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(hold);

    // checkpoint, clobber, restart and verify the random data on every rank
    int rc = checkpoint_restart_verify(regs, "test_multigpu_alloc", 1);
    if (rc != 0)
        printf("RANK %d: checkpoint/restart verification failed (%d)\n", rank, rc);
    int total_rc = 0;
    MPI_Allreduce(&rc, &total_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    VELOC_Finalize(1);
    free_regions(regs);
    MPI_Finalize();
    return total_rc;
}
