#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <thread>
#include <vector>
#include <string>

#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>

#include "include/veloc.h"

/*
    Host-only regression tests for the checkpoint completion semantics of the
    client/transfer-engine pipeline. Each case targets one invariant:

      fail_not_flushed  A checkpoint whose local write failed must not reach the
                        backend: an older good version must stay restorable.
      route_file        File-based (Route_file) checkpoints must be handed to
                        the backend in async mode (Checkpoint_finished turns true).
      route_file_verify Second phase of route_file, run after the script removed
                        scratch: the checkpoint must be restorable from persistent.
      finalize_nodrain  Finalize(0) right after an async checkpoint must not crash
                        and must still hand the checkpoint to the backend.
      finalize_verify   Second phase of finalize_nodrain (scratch removed).
      drain             Only Init + Finalize(1): waits for earlier commands.
      finished_after_flush  Poll Checkpoint_finished() without Checkpoint_wait():
                        once it is true, the persistent copy must be complete
                        (needs the persistent directory as 3rd argument).
      wait_finished     After Checkpoint_wait() succeeds, Checkpoint_finished()
                        must already be true (async). Probabilistic: many tiny
                        checkpoints to hit the window between "group done" and
                        "handed to backend".

    Usage: test_ckpt_semantics <cfg_file> <case> [iterations] [size_MiB]
           test_ckpt_semantics <cfg_file> finished_after_flush <persistent_dir>
    Exit: 0 pass, 1 invariant violated, 2 setup error, 77 not applicable.
*/

#define NAME_MEM  "semantics_mem"
#define NAME_FILE "semantics_file"

static const size_t REGION_BYTES = 64UL << 20;

static void fill(unsigned char *p, size_t n, uint64_t seed) {
    uint64_t s = seed;
    for (size_t i = 0; i < n; i += 8) {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z ^= z >> 31;
        memcpy(p + i, &z, n - i < 8 ? n - i : 8);
    }
}

static bool matches(const unsigned char *p, size_t n, uint64_t seed) {
    std::vector<unsigned char> expect(n);
    fill(expect.data(), n, seed);
    return memcmp(expect.data(), p, n) == 0;
}

#define FAIL(...) do { printf("FAIL: " __VA_ARGS__); printf("\n"); return 1; } while (0)
#define SETUP(call) do { if ((call) != VELOC_SUCCESS) { printf("setup error: %s\n", #call); return 2; } } while (0)

static bool async_mode(const char *cfg) {
    FILE *f = fopen(cfg, "r");
    if (f == NULL)
        return false;
    char line[256];
    bool async = false;
    while (fgets(line, sizeof(line), f))
        if (strstr(line, "mode") && strstr(line, "async"))
            async = true;
    fclose(f);
    return async;
}

static int checkpoint_mem_version(const char *name, int v) {
    if (VELOC_Checkpoint_begin(name, v) != VELOC_SUCCESS) return 0;
    if (VELOC_Checkpoint_mem() != VELOC_SUCCESS) return 0;
    if (VELOC_Checkpoint_end(1) != VELOC_SUCCESS) return 0;
    return VELOC_Checkpoint_wait() == VELOC_SUCCESS;
}

// Finding 1: v1 is good; v2's local write fails (RLIMIT_FSIZE makes pwrite past
// the limit return EFBIG). With max_versions = 1, sending v2 to the backend
// deletes v1 and leaves only the truncated v2.
static int case_fail_not_flushed() {
    std::vector<unsigned char> buf(REGION_BYTES);
    SETUP(VELOC_Mem_protect(0, buf.data(), buf.size(), 1));

    fill(buf.data(), buf.size(), 1);
    if (!checkpoint_mem_version(NAME_MEM, 1))
        FAIL("v1 checkpoint failed, cannot run the test");

    fill(buf.data(), buf.size(), 2);
    struct rlimit old_lim, lim;
    getrlimit(RLIMIT_FSIZE, &old_lim);
    lim = old_lim;
    lim.rlim_cur = REGION_BYTES / 8;
    signal(SIGXFSZ, SIG_IGN); // get EFBIG instead of being killed
    if (setrlimit(RLIMIT_FSIZE, &lim) != 0) {
        printf("cannot set RLIMIT_FSIZE\n");
        return 2;
    }
    bool reported = VELOC_Checkpoint_begin(NAME_MEM, 2) != VELOC_SUCCESS;
    reported |= VELOC_Checkpoint_mem() != VELOC_SUCCESS;
    reported |= VELOC_Checkpoint_end(1) != VELOC_SUCCESS;
    reported |= VELOC_Checkpoint_wait() != VELOC_SUCCESS;
    setrlimit(RLIMIT_FSIZE, &old_lim);

    if (!reported)
        FAIL("v2 write was truncated but no call reported a failure");
    int latest = VELOC_Restart_test(NAME_MEM, 0);
    if (latest != 1)
        FAIL("latest restorable version is %d, expected 1 (failed v2 was sent to the backend)", latest);
    memset(buf.data(), 0, buf.size());
    if (VELOC_Restart(NAME_MEM, 1) != VELOC_SUCCESS)
        FAIL("cannot restart from v1 (it was deleted after the failed v2)");
    if (!matches(buf.data(), buf.size(), 1))
        FAIL("restored v1 content differs");
    printf("failed checkpoint was kept local, v1 still restorable\n");
    return 0;
}

static bool write_route_file(int v) {
    char original[VELOC_MAX_NAME], routed[VELOC_MAX_NAME];
    snprintf(original, sizeof(original), "semantics_route_%d.dat", v);
    if (VELOC_Route_file(original, routed) != VELOC_SUCCESS)
        return false;
    std::vector<unsigned char> data(4UL << 20);
    fill(data.data(), data.size(), 100 + v);
    FILE *f = fopen(routed, "wb");
    if (f == NULL)
        return false;
    bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}

// Finding 2: Checkpoint_end on a file-based checkpoint (no Checkpoint_mem)
// must still hand it to the backend.
static int case_route_file(bool async) {
    if (!async) {
        printf("route_file applies to async mode only\n");
        return 77;
    }
    for (int v = 1; v <= 2; v++) {
        SETUP(VELOC_Checkpoint_wait());
        SETUP(VELOC_Checkpoint_begin(NAME_FILE, v));
        bool ok = write_route_file(v);
        SETUP(VELOC_Checkpoint_end(ok));
        if (!ok)
            FAIL("cannot write routed file for v%d", v);
    }
    if (VELOC_Checkpoint_wait() != VELOC_SUCCESS)
        FAIL("Checkpoint_wait reported failure");
    for (int i = 0; i < 300 && VELOC_Checkpoint_finished() != VELOC_SUCCESS; i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    if (VELOC_Checkpoint_finished() != VELOC_SUCCESS)
        FAIL("Checkpoint_finished still false 30 s after Checkpoint_wait succeeded (never handed to backend)");
    printf("file-based checkpoints handed to the backend\n");
    return 0;
}

static int case_route_file_verify() {
    int v = VELOC_Restart_test(NAME_FILE, 0);
    if (v != 2)
        FAIL("latest persistent version is %d, expected 2 (file checkpoints were not flushed)", v);
    SETUP(VELOC_Restart_begin(NAME_FILE, v));
    char original[VELOC_MAX_NAME], routed[VELOC_MAX_NAME];
    snprintf(original, sizeof(original), "semantics_route_%d.dat", v);
    SETUP(VELOC_Route_file(original, routed));
    std::vector<unsigned char> data(4UL << 20);
    FILE *f = fopen(routed, "rb");
    bool ok = f != NULL && fread(data.data(), 1, data.size(), f) == data.size();
    if (f) fclose(f);
    SETUP(VELOC_Restart_end(ok));
    if (!ok || !matches(data.data(), data.size(), 100 + v))
        FAIL("restored file content differs");
    printf("file-based checkpoint restored from persistent\n");
    return 0;
}

// Finding 3: Finalize(0) while the local stage is still in flight. The process
// must exit cleanly and the checkpoint must still reach the backend.
static int case_finalize_nodrain(bool async) {
    if (!async) {
        printf("finalize_nodrain applies to async mode only\n");
        return 77;
    }
    std::vector<unsigned char> buf(8 * REGION_BYTES); // big enough to still be draining
    SETUP(VELOC_Mem_protect(0, buf.data(), buf.size(), 1));
    fill(buf.data(), buf.size(), 7);
    SETUP(VELOC_Checkpoint_begin(NAME_MEM, 1));
    SETUP(VELOC_Checkpoint_mem());
    SETUP(VELOC_Checkpoint_end(1));
    VELOC_Finalize(0);
    printf("Finalize(0) returned\n");
    return 0; // a crash in the engine thread shows up as a non-zero exit
}

static int case_finalize_verify() {
    int v = VELOC_Restart_test(NAME_MEM, 0);
    if (v != 1)
        FAIL("latest persistent version is %d, expected 1 (checkpoint lost by Finalize(0))", v);
    std::vector<unsigned char> buf(8 * REGION_BYTES);
    SETUP(VELOC_Mem_protect(0, buf.data(), buf.size(), 1));
    if (VELOC_Restart(NAME_MEM, 1) != VELOC_SUCCESS)
        FAIL("cannot restart v1");
    if (!matches(buf.data(), buf.size(), 7))
        FAIL("restored content differs");
    printf("checkpoint written before Finalize(0) restored from persistent\n");
    return 0;
}

// Finding 4: Checkpoint_wait() must not return before the checkpoint is queued
// at the backend, i.e. Checkpoint_finished() must hold right after it.
static int case_wait_finished(bool async, int iters, size_t bytes) {
    if (!async) {
        printf("wait_finished applies to async mode only\n");
        return 77;
    }
    // the group must still be draining when Checkpoint_wait() starts, otherwise
    // the continuation runs inline in Checkpoint_end() and there is no race
    std::vector<unsigned char> buf(bytes);
    SETUP(VELOC_Mem_protect(0, buf.data(), buf.size(), 1));
    int violations = 0;
    for (int i = 1; i <= iters; i++) {
        fill(buf.data(), buf.size(), i);
        if (!checkpoint_mem_version(NAME_MEM, i))
            FAIL("checkpoint %d failed", i);
        if (VELOC_Checkpoint_finished() != VELOC_SUCCESS)
            violations++;
    }
    if (violations > 0)
        FAIL("%d of %d Checkpoint_wait() calls returned before the checkpoint was handed to the backend", violations, iters);
    printf("Checkpoint_finished held after every Checkpoint_wait (%d iterations)\n", iters);
    return 0;
}

// Checkpoint_finished() means "no longer happening on the backend": it must
// not turn true once the local stage is done but the flush is still running.
static int case_finished_after_flush(bool async, const char *persistent) {
    if (!async) {
        printf("finished_after_flush applies to async mode only\n");
        return 77;
    }
    std::vector<unsigned char> buf(8 * REGION_BYTES); // flush takes long enough to poll
    SETUP(VELOC_Mem_protect(0, buf.data(), buf.size(), 1));
    fill(buf.data(), buf.size(), 9);
    SETUP(VELOC_Checkpoint_begin(NAME_MEM, 1));
    SETUP(VELOC_Checkpoint_mem());
    SETUP(VELOC_Checkpoint_end(1));

    auto t0 = std::chrono::steady_clock::now();
    while (VELOC_Checkpoint_finished() != VELOC_SUCCESS) {
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(60))
            FAIL("Checkpoint_finished still false after 60 s");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // the persistent copy must be complete right now, not eventually
    const size_t header = sizeof(size_t) + sizeof(int) + sizeof(size_t);
    std::string fname = std::string(persistent) + "/" + NAME_MEM + "-0-1.dat";
    struct stat st;
    if (stat(fname.c_str(), &st) != 0)
        FAIL("Checkpoint_finished true but %s does not exist yet", fname.c_str());
    if ((size_t)st.st_size != header + buf.size())
        FAIL("Checkpoint_finished true but %s has %zu of %zu bytes (backend still flushing)",
             fname.c_str(), (size_t)st.st_size, header + buf.size());
    std::vector<unsigned char> data(buf.size());
    FILE *f = fopen(fname.c_str(), "rb");
    bool ok = f != NULL && fseek(f, header, SEEK_SET) == 0 && fread(data.data(), 1, data.size(), f) == data.size();
    if (f) fclose(f);
    if (!ok || !matches(data.data(), data.size(), 9))
        FAIL("Checkpoint_finished true but the persistent content is not the checkpoint yet");
    printf("Checkpoint_finished turned true only once the persistent copy was complete\n");
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <cfg_file> <case> [iterations] [size_MiB]\n", argv[0]);
        return 2;
    }
    const char *cfg = argv[1];
    std::string c = argv[2];
    int iters = argc > 3 ? atoi(argv[3]) : 500;
    bool async = async_mode(cfg);

    if (VELOC_Init_single(0, cfg) != VELOC_SUCCESS) {
        printf("Error initializing VELOC\n");
        return 2;
    }
    int rc;
    if (c == "fail_not_flushed")       rc = case_fail_not_flushed();
    else if (c == "route_file")        rc = case_route_file(async);
    else if (c == "route_file_verify") rc = case_route_file_verify();
    else if (c == "finalize_nodrain")  return case_finalize_nodrain(async); // already finalized
    else if (c == "finalize_verify")   rc = case_finalize_verify();
    else if (c == "wait_finished")     rc = case_wait_finished(async, iters, (argc > 4 ? strtoul(argv[4], NULL, 10) : 1UL) << 20);
    else if (c == "drain")             rc = 0; // Init already waited for earlier commands
    else if (c == "finished_after_flush") rc = argc > 3 ? case_finished_after_flush(async, argv[3]) : 2;
    else {
        printf("unknown case %s\n", c.c_str());
        rc = 2;
    }
    VELOC_Finalize(1);
    return rc;
}
