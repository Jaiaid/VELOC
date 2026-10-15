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
    void *d_data = NULL;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, rank + 1, bytes));
    VELOC_Mem_protect(0, d_data, bytes, 1);

    printf("RANK %d PID %d GPU %d READY\n", rank, (int)getpid(), dev);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(hold);

    int rc = 0;
    if (VELOC_Checkpoint("test_multigpu_alloc", 1) != VELOC_SUCCESS) {
        printf("RANK %d: checkpoint failed\n", rank);
        rc = 6;
    }
    VELOC_Checkpoint_wait();
    int total_rc = 0;
    MPI_Allreduce(&rc, &total_rc, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    VELOC_Finalize(1);
    cudaFree(d_data);
    MPI_Finalize();
    return total_rc;
}
