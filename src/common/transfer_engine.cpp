#include "transfer_engine.hpp"
#include "common/config.hpp"
#include "file_provider.hpp"
#include "common/tiers/mem_provider.hpp"

#include <numa.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <algorithm>
#include <map>
#include <utility>
#include <cstring>
#include <cstdlib>
#include <unistd.h>

//#define __DEBUG
#include "common/debug.hpp"

// Compile-time tunables (overridable via -D).
#ifndef VELOC_XFER_CHUNK
#define VELOC_XFER_CHUNK (64UL << 20)   // 64 MiB per staged chunk
#endif
#ifndef VELOC_GPU_STAGE_SLOTS
#define VELOC_GPU_STAGE_SLOTS 8        // bounded pinned pool: SLOTS * CHUNK bytes
#endif
#ifndef VELOC_HOST_STAGE_SLOTS
#define VELOC_HOST_STAGE_SLOTS 16        // bounded pinned pool: SLOTS * CHUNK bytes
#endif

// Fallbacks when no GPU support is compiled in: everything is host memory.
#ifndef WITH_CUDA
int gpu_device_count() { return 0; }
int gpu_device_of(const void *) { return -1; }
gpu_provider_t *create_gpu_provider(int, size_t, int) { return nullptr; }
#endif

namespace {
enum kind_t { K_HOST, K_DEVICE, K_FILE };
struct dchunk_t { void *dev; int fd; size_t foff; size_t n; };

// classify() result: T_FILE for file endpoints, otherwise the memory tier
// (mem_dev_t) packed into an int as (type << 16 | id).
const int T_FILE = -1;
inline int encode_tier(mem_dev_t d) { return (d.type << 16) | (d.id & 0xffff); }
inline mem_dev_t decode_tier(int t) { return mem_dev_t(t >> 16, t & 0xffff); }
inline int kind_of(int t) {
    if (t == T_FILE)
        return K_FILE;
    return decode_tier(t).type == DEV_GPU ? K_DEVICE : K_HOST;
}
}

struct xfer_t { endpoint_t from, to; size_t len; };

struct xfer_group_impl_t {
    transfer_engine_t::impl_t *eng = nullptr;
    std::vector<xfer_t> transfers;
    std::vector<std::unique_ptr<char[]>> buffers; // header/serialized data, alive until group destroyed
    std::vector<int> fds;                         // closed by the engine when durable

    size_t src_total = 0, src_done = 0;
    size_t all_total = 0, all_done = 0;
    bool queued = false, work_done = false, completion_ran = false, failed = false;
    std::function<void()> completion;

    ~xfer_group_impl_t() {
        for (int fd : fds)
            if (fd >= 0)
                ::close(fd);
    }
};

class transfer_engine_t::impl_t {
private:
    // Discover every memory tier on the node: one host_provider_t per NUMA node,
    // then one gpu_provider_t per GPU. Only the providers are created here; their
    // staging pools are allocated lazily by init_tiers_on_path().
    void discover_tiers()
    {
        int num_host_stage_slots = VELOC_HOST_STAGE_SLOTS;
        cfg.get_optional<int>("host_stage_slots", num_host_stage_slots);
        int num_gpu_stage_slots = VELOC_GPU_STAGE_SLOTS;
        cfg.get_optional<int>("per-gpu_stage_slots", num_gpu_stage_slots);

        if (numa_available() < 0) {
            INFO("libnuma unavailable, using a single host tier");
            tiers.push_back(new host_provider_t(0, chunk, num_host_stage_slots));
        } else {
            for (int n = 0; n <= numa_max_node(); n++)
                if (numa_bitmask_isbitset(numa_all_nodes_ptr, n))
                    tiers.push_back(new host_provider_t(n, chunk, num_host_stage_slots));
        }

        int gpus = gpu_device_count();
        for (int d = 0; d < gpus; d++) {
            gpu_provider_t *gpu = create_gpu_provider(d, chunk, num_gpu_stage_slots);
            if (gpu != nullptr)
                tiers.push_back(gpu);
        }
        INFO(
            "discovered " << tiers.size() - gpus << " NUMA host tier(s) and " << gpus << " GPU tier(s)" << std::endl
        );
    }

    // NUMA node holding the page of ptr, or -1 if the page is not yet faulted in.
    static int numa_node_of(const void *ptr) {
        if (numa_available() < 0)
            return 0;
        void *page = (void *)((uintptr_t)ptr & ~((uintptr_t)numa_pagesize() - 1));
        int status = -1;
        if (numa_move_pages(0, 1, &page, nullptr, &status, 0) != 0 || status < 0)
            return -1;
        return status;
    }

    bool init_tier_buffer(mem_provider_t *t) {
        if (t->buffer_initialized())
            return true;
        mem_dev_t d = t->get_device_type();
        INFO(
            "Initiating " << (d.type == DEV_GPU ? "device staging pool on GPU " : "host staging pool on NUMA node ") <<
            d.id << ": " << t->slot_size() << " byte slots" << std::endl
        );
        if (!t->init_buffer()) {
            ERROR("failed to initialize staging buffer on " << (d.type == DEV_GPU ? "GPU " : "NUMA node ") << d.id);
            return false;
        }
        return true;
    }

public:
    std::vector<mem_provider_t *> tiers; // every memory tier on the node, host tiers first
    size_t chunk = VELOC_XFER_CHUNK;

    const config_t &cfg; // reference to the config object used to initialize different tiers of the transfer engine

    std::mutex mtx;
    std::condition_variable app_cv;  // application waiters (wait_sources/wait_completion)
    std::condition_variable work_cv; // progress thread
    std::deque<std::shared_ptr<xfer_group_impl_t>> ready;
    std::thread worker;
    bool stop = false;

    impl_t(const config_t &cfg) : cfg(cfg) {
        // All tiers are discovered up front, but no staging memory is allocated
        // until a protected pointer needs a tier (see init_tiers_on_path()), so
        // that unused NUMA nodes and GPUs cost nothing.
        // If not a GPU build, gpu_device_count() returns 0 and only host tiers exist.
        discover_tiers();
        worker = std::thread([this] { run(); });
    }

    ~impl_t() {
        {
            std::unique_lock<std::mutex> lk(mtx);
            stop = true;
            work_cv.notify_all();
        }
        if (worker.joinable())
            worker.join();
        for (mem_provider_t *t : tiers)
            delete t;
    }

    // Returns T_FILE, or the encoded tier (see encode_tier()) the memory belongs to:
    // the owning GPU for device memory, otherwise the NUMA node holding it.
    int classify(const endpoint_t &e) {
        if (e.fd >= 0)
            return T_FILE;
        int gpu_id = gpu_device_of(e.ptr);
        if (gpu_id >= 0)
            return encode_tier(mem_dev_t(DEV_GPU, gpu_id));
        int node = numa_node_of(e.ptr);
        if (node < 0) // untouched page: attribute to the first host tier
            node = tiers.front()->get_device_type().id;
        return encode_tier(mem_dev_t(DEV_CPU, node));
    }

    // Search the tier list for the provider of device d, nullptr if none.
    mem_provider_t *find_tier(mem_dev_t d) {
        for (mem_provider_t *t : tiers)
            if (t->get_device_type() == d)
                return t;
        return nullptr;
    }

    gpu_provider_t *find_gpu(int device_id) {
        return static_cast<gpu_provider_t *>(find_tier(mem_dev_t(DEV_GPU, device_id)));
    }

    // Host tier used to stage data of the given GPU on its way to/from a file:
    // the GPU's affiliated NUMA node, or the first host tier if that is unknown.
    mem_provider_t *staging_host_of(const gpu_provider_t *gpu) {
        mem_provider_t *host = find_tier(mem_dev_t(DEV_CPU, gpu->affiliated_numa_node()));
        return host != nullptr ? host : tiers.front();
    }

    // This method is kept public for deferred initialization of the tiers, since only the
    // protected pointer tells us which GPU / NUMA node will be used.
    // It is called from transfer_engine_t::init_tier() on every mem_protect().
    // As we are using Pimpl Idiom pattern this is called inside transfer_engin_t::init_tier and needs to be public
    // Allocates the staging pools of every tier on the pointer's path to a file:
    // a device pointer goes GPU pool -> affiliated NUMA host pool -> file, while a host
    // pointer is written directly and needs no pool. Already initialized tiers are skipped.
    void init_tiers_on_path(void *ptr) {
        int t = classify(mem(ptr));
        if (kind_of(t) != K_DEVICE)
            return;
        gpu_provider_t *gpu = find_gpu(decode_tier(t).id);
        if (gpu == nullptr) {
            ERROR("no tier for GPU " << decode_tier(t).id);
            return;
        }
        init_tier_buffer(staging_host_of(gpu));
        init_tier_buffer(gpu);
    }

    void ensure_queued_locked(const std::shared_ptr<xfer_group_impl_t> &g) {
        if (!g->queued) {
            g->queued = true;
            ready.push_back(g);
            work_cv.notify_one();
        }
    }

    void add_progress(const std::shared_ptr<xfer_group_impl_t> &g, size_t dsrc, size_t dall) {
        std::unique_lock<std::mutex> lk(mtx);
        g->src_done += dsrc;
        g->all_done += dall;
        app_cv.notify_all();
    }

    void fail(const std::shared_ptr<xfer_group_impl_t> &g) {
        std::unique_lock<std::mutex> lk(mtx);
        g->failed = true;
    }

    void run() {
        while (true) {
            std::shared_ptr<xfer_group_impl_t> g;
            {
                std::unique_lock<std::mutex> lk(mtx);
                work_cv.wait(lk, [&] { return stop || !ready.empty(); });
                if (ready.empty()) {
                    if (stop)
                        break;
                    continue;
                }
                g = ready.front();
                ready.pop_front();
            }
            process(g);
        }
    }

    // All chunks must belong to the same GPU (process() groups them per GPU).
    void run_device_out(const std::shared_ptr<xfer_group_impl_t> &g, std::vector<dchunk_t> &chunks) {
        // Three-tier device->file drain. The application's device memory is
        // first snapshotted with a fast D2D into the device staging pool; when
        // that pool is full, the oldest device-staged chunk is evicted (D2H)
        // into a pinned slot of the GPU's affiliated NUMA node and the slot is
        // reused; when the pinned pool is also full, the oldest pinned chunk is
        // written to the file. This keeps the device pool "hot" (incoming chunks
        // keep landing on-device) and guarantees the file is written in capture
        // order, since pin_staged always holds strictly older data than dev_staged.
        gpu_provider_t *gpu = find_gpu(decode_tier(classify(mem(chunks.front().dev))).id);
        if (gpu == nullptr) {
            ERROR("no tier for GPU owning " << chunks.front().dev);
            fail(g);
            return;
        }
        mem_provider_t *host = staging_host_of(gpu);
        if (!host->buffer_initialized()) {
            ERROR("no host staging pool for GPU " << gpu->get_device_type().id << ", was the region protected?");
            fail(g);
            return;
        }

        std::deque<std::pair<void *, dchunk_t>> dev_staged; // device slots holding app data
        std::deque<std::pair<void *, dchunk_t>> pin_staged; // pinned slots awaiting file write

        // Return every held slot to its pool (used on failure so later groups
        // are not starved; data in these slots is not written to the file).
        auto release_all = [&]() {
            for (auto &p : dev_staged)
                gpu->release_slot(p.first);
            dev_staged.clear();
            for (auto &p : pin_staged)
                host->release_slot(p.first);
            pin_staged.clear();
        };

        // Evict the oldest device-staged chunk into a free pinned slot. The
        // device slot is released either way; returns false on copy failure.
        auto move_dev_to_pin = [&]() -> bool {
            void *pin = host->acquire_slot();
            void *dev = dev_staged.front().first;
            if (!gpu->copy_d2h(pin, dev, dev_staged.front().second.n)) {
                fail(g);
                gpu->release_slot(dev);
                host->release_slot(pin);
                dev_staged.pop_front();
                return false;
            }
            pin_staged.push_back({pin, dev_staged.front().second});
            gpu->release_slot(dev);
            dev_staged.pop_front();
            return true;
        };

        // Write the oldest pinned chunk to the file and return the slot.
        auto drain_pin_to_file = [&]() {
            void *pin = pin_staged.front().first;
            const dchunk_t &c = pin_staged.front().second;
            if (!file_provider::write_range(c.fd, c.foff, pin, c.n))
                fail(g);
            else
                add_progress(g, 0, c.n);
            host->release_slot(pin);
            pin_staged.pop_front();
        };

        size_t i = 0;
        while (i < chunks.size() && !g->failed) {
            const dchunk_t &c = chunks[i];
            void *devptr = gpu->acquire_slot();
            if (devptr != nullptr) {
                if (!gpu->copy_d2d(devptr, c.dev, c.n)) {
                    fail(g);
                    gpu->release_slot(devptr);
                    break;
                }
                add_progress(g, c.n, 0); // application device memory released
                dev_staged.push_back({devptr, c});
                i++;
                continue;
            }
            if (!dev_staged.empty() && host->free_slots() > 0) {
                if (!move_dev_to_pin()) // frees a device slot
                    break;
                continue; // stage into the freed device slot
            }
            if (host->free_slots() > 0) {
                // No device pool in use (init failed): stage into pinned.
                void *pin = host->acquire_slot();
                if (!gpu->copy_d2h(pin, c.dev, c.n)) {
                    fail(g);
                    host->release_slot(pin);
                    break;
                }
                add_progress(g, c.n, 0);
                pin_staged.push_back({pin, c});
                i++;
                continue;
            }
            // Both pools exhausted: flush the oldest pinned chunk to make room.
            // We only free one pinned slot at a time, this is done so that iterations are faster 
            // if the pinned pool is not full, and to avoid starving the device pool if it is full. 
            drain_pin_to_file();
        }

        // Flush remaining tiers to the file, oldest-first.
        while (!g->failed) {
            if (!dev_staged.empty()) {
                if (host->free_slots() == 0) {
                    if (pin_staged.empty()) {
                        ERROR("no pinned slot available when flushing device staging");
                        fail(g);
                        break;
                    }
                    drain_pin_to_file();
                } else if (!move_dev_to_pin()) {
                    break;
                }
            } else if (!pin_staged.empty()) {
                drain_pin_to_file();
            } else {
                break;
            }
        }

        release_all();
    }

    void process(const std::shared_ptr<xfer_group_impl_t> &g) {
        std::map<int, std::vector<dchunk_t>> dev_out; // per source GPU tier
        for (auto &t : g->transfers) {
            if (g->failed)
                break;
            int tf = classify(t.from), tt = classify(t.to);
            int kf = kind_of(tf), kt = kind_of(tt);
            if (kf == K_HOST && kt == K_FILE) {
                if (!file_provider::write_range(t.to.fd, t.to.off, t.from.ptr, t.len))
                    fail(g);
                else
                    add_progress(g, t.len, t.len);
            } else if (kf == K_DEVICE && kt == K_FILE) {
                for (size_t o = 0; o < t.len; o += chunk) {
                    size_t n = std::min(chunk, t.len - o);
                    dev_out[tf].push_back({(char *)t.from.ptr + o, t.to.fd, t.to.off + o, n});
                }
            } else if (kf == K_FILE && kt == K_HOST) {
                if (!file_provider::read_range(t.from.fd, t.from.off, t.to.ptr, t.len))
                    fail(g);
                else
                    add_progress(g, t.len, t.len);
            } else if (kf == K_FILE && kt == K_DEVICE) {
                gpu_provider_t *gpu = find_gpu(decode_tier(tt).id);
                mem_provider_t *host = gpu != nullptr ? staging_host_of(gpu) : nullptr;
                for (size_t o = 0; o < t.len && !g->failed; o += chunk) {
                    size_t n = std::min(chunk, t.len - o);
                    void *slot = host != nullptr ? host->acquire_slot() : nullptr;
                    if (slot == nullptr) {
                        ERROR("no staging slot available for device restore");
                        fail(g);
                        break;
                    }
                    if (!file_provider::read_range(t.from.fd, t.from.off + o, slot, n) ||
                        !gpu->copy_h2d((char *)t.to.ptr + o, slot, n))
                        fail(g);
                    else
                        add_progress(g, n, n);
                    host->release_slot(slot);
                }
            } else if (kf == K_FILE && kt == K_FILE) {
                if (!file_provider::copy_range(t.from.fd, t.from.off, t.to.fd, t.to.off, t.len))
                    fail(g);
                else
                    add_progress(g, t.len, t.len);
            } else if (kf == K_HOST && kt == K_HOST) {
                memcpy(t.to.ptr, t.from.ptr, t.len);
                add_progress(g, t.len, t.len);
            } else {
                ERROR("unsupported transfer between endpoint kinds " << kf << " and " << kt);
                fail(g);
            }
        }
        for (auto &d : dev_out) {
            if (g->failed)
                break;
            run_device_out(g, d.second);
        }
        finish(g);
    }

    void finish(const std::shared_ptr<xfer_group_impl_t> &g) {
        std::function<void()> cb;
        {
            std::unique_lock<std::mutex> lk(mtx);
            g->src_done = g->src_total; // ensure waiters wake even after an early failure
            g->all_done = g->all_total;
            g->work_done = true;
            for (int fd : g->fds)
                if (fd >= 0)
                    ::close(fd);
            g->fds.clear();
            if (g->completion && !g->completion_ran) {
                cb = g->completion;
                g->completion_ran = true;
            }
        }
        if (cb)
            cb();
        std::unique_lock<std::mutex> lk(mtx);
        app_cv.notify_all();
    }
};

// ---- xfer_group_t ----------------------------------------------------------

void xfer_group_t::submit(endpoint_t from, endpoint_t to, size_t len) {
    if (!g || len == 0)
        return;
    g->transfers.push_back({from, to, len});
    g->src_total += len;
    g->all_total += len;
}

void *xfer_group_t::alloc(size_t size) {
    if (!g)
        return nullptr;
    g->buffers.emplace_back(new char[size]);
    return g->buffers.back().get();
}

void xfer_group_t::adopt_fd(int fd) {
    if (g)
        g->fds.push_back(fd);
}

bool xfer_group_t::wait_sources() {
    if (!g)
        return true;
    auto *e = g->eng;
    std::unique_lock<std::mutex> lk(e->mtx);
    e->ensure_queued_locked(g);
    e->app_cv.wait(lk, [&] { return g->src_done >= g->src_total; });
    return !g->failed;
}

bool xfer_group_t::wait_completion() {
    if (!g)
        return true;
    auto *e = g->eng;
    std::function<void()> cb;
    {
        std::unique_lock<std::mutex> lk(e->mtx);
        e->ensure_queued_locked(g);
        e->app_cv.wait(lk, [&] { return g->work_done; });
        if (g->completion && !g->completion_ran) {
            cb = g->completion;
            g->completion_ran = true;
        }
    }
    if (cb)
        cb();
    return !g->failed;
}

void xfer_group_t::on_completion(std::function<void()> cont) {
    if (!g)
        return;
    auto *e = g->eng;
    std::function<void()> run_now;
    {
        std::unique_lock<std::mutex> lk(e->mtx);
        if (g->work_done && !g->completion_ran) {
            g->completion_ran = true;
            run_now = std::move(cont);
        } else
            g->completion = std::move(cont);
    }
    if (run_now)
        run_now();
}

// ---- transfer_engine_t -----------------------------------------------------

transfer_engine_t::transfer_engine_t(const config_t &cfg) : pimpl(new impl_t(cfg)) { }
transfer_engine_t::~transfer_engine_t() = default;

transfer_engine_t &transfer_engine_t::instance(const config_t &cfg) {
    static transfer_engine_t engine(cfg);
    return engine;
}

xfer_group_t transfer_engine_t::group() {
    auto g = std::make_shared<xfer_group_impl_t>();
    g->eng = pimpl.get();
    return xfer_group_t(g);
}

void transfer_engine_t::init_tier(void *ptr) {
    pimpl->init_tiers_on_path(ptr);
}
