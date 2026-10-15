#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>

#include "gpu_test_util.h"
#include "include/veloc.h"

/*
    MPI version of heatdis_gpu (see build/test/test_gpu.sh). Each rank owns a
    device array on GPU (rank % num_gpus) plus a host iteration counter. Run it
    twice with the same config: the first run checkpoints, the second run finds
    the checkpoint, restarts and verifies the device contents on every rank.

    Usage: mpirun -np N test_gpu_mpi <cfg_file> [array_MiB=256]
*/

static double expected(size_t i, int iter, int rank) {
    return (double)((i + (size_t)iter + (size_t)rank) % 7);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <cfg_file> [array_MiB]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    size_t n = ((argc > 2 ? strtoul(argv[2], NULL, 10) : 256UL) << 20) / sizeof(double);

    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    CUDA_CHECK(cudaSetDevice(rank % ngpu));

    if (VELOC_Init(MPI_COMM_WORLD, argv[1]) != VELOC_SUCCESS) {
        printf("Error initializing VELOC! Aborting...\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int iter = 0, rc = 0;
    double *d_data = NULL;
    double *h = (double *)malloc(n * sizeof(double));
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(double)));
    VELOC_Mem_protect(0, &iter, 1, sizeof(int));
    VELOC_Mem_protect(1, d_data, n, sizeof(double));

    int v = VELOC_Restart_test("test_gpu_mpi", 0);
    if (v > 0) {
        if (rank == 0) printf("Previous checkpoint found (version %d), restarting...\n", v);
        if (VELOC_Restart("test_gpu_mpi", v) != VELOC_SUCCESS) {
            printf("RANK %d: error restarting!\n", rank);
            rc = 4;
        } else {
            CUDA_CHECK(cudaMemcpy(h, d_data, n * sizeof(double), cudaMemcpyDeviceToHost));
            int ok = (iter == 3);
            for (size_t i = 0; ok && i < n; i++)
                if (h[i] != expected(i, iter, rank)) {
                    printf("RANK %d: MISMATCH at %zu: got %f expected %f\n", rank, i, h[i], expected(i, iter, rank));
                    ok = 0;
                }
            printf("RANK %d (GPU %d): restart verification %s (iter=%d)\n", rank, rank % ngpu, ok ? "OK" : "FAIL", iter);
            rc = ok ? 0 : 5;
        }
    } else {
        iter = 3;
        for (size_t i = 0; i < n; i++)
            h[i] = expected(i, iter, rank);
        CUDA_CHECK(cudaMemcpy(d_data, h, n * sizeof(double), cudaMemcpyHostToDevice));
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
    cudaFree(d_data);
    free(h);
    MPI_Finalize();
    return total_rc;
}
