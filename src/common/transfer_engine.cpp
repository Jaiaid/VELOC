#include "transfer_engine.hpp"
#include "common/config.hpp"
#include "file_provider.hpp"
#include "common/gpu/gpu_provider.hpp"

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <algorithm>
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

// Fallback factory when no GPU support is compiled in: everything is host memory.
#ifndef WITH_CUDA
gpu_provider_t *create_gpu_provider() { return nullptr; }
#endif

namespace {
enum kind_t { K_HOST, K_DEVICE, K_FILE };
struct dchunk_t { void *dev; int fd; size_t foff; size_t n; };
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
    void init_host_tier(const config_t &cfg)
    {
        int num_host_stage_slots = VELOC_HOST_STAGE_SLOTS;
        cfg.get_optional<int>("host_stage_slots", num_host_stage_slots);
        INFO(
            "Initiating host staging pool: " << num_host_stage_slots << " slots of " << 
            chunk << std::endl
        );

        for (int i = 0; i < num_host_stage_slots; i++) {
            void *s;
            bool pinned;
            if (gpu != nullptr) {
                s = gpu->alloc_pinned(chunk);
                pinned = (s != nullptr);
                if (s == nullptr)
                    s = ::malloc(chunk);
                if (s == nullptr)
                    break;
            }
            else {
                s = ::malloc(chunk);
                pinned = false;
                if (s == nullptr)
                    break;
            }

            slots_all.push_back(s);
            slots_pinned.push_back(pinned);
            free_slots.push_back(s);
        }
    }

public:
    gpu_provider_t *gpu = nullptr;
    size_t chunk = VELOC_XFER_CHUNK;

    const config_t &cfg; // reference to the config object used to initialize different tiers of the transfer engine

    std::vector<void *> slots_all;   // every allocated slot (for teardown)
    std::vector<bool> slots_pinned;
    std::vector<void *> free_slots;  // touched only by the worker thread

    std::mutex mtx;
    std::condition_variable app_cv;  // application waiters (wait_sources/wait_completion)
    std::condition_variable work_cv; // progress thread
    std::deque<std::shared_ptr<xfer_group_impl_t>> ready;
    std::thread worker;
    bool stop = false;

    impl_t(const config_t &cfg) : cfg(cfg) {
        // We assume that host tier is always available, so we create the host pool in constructor rather then deferring. 
        // The GPU tier is optional and will be created only if GPU support is compiled in and a device pointer is seen.
        // it can be pinned or unpinned depending on whether GPU support is available and pinned memory allocation succeeds.
        init_host_tier(cfg);
        // we create GPU provider but the buffer allocation at GPU tier is deferred until the first device pointer is seen, 
        // so that we can choose the right device for multi-GPU systems
        // also if not GPU build, create_gpu_provider() will return nullptr and the GPU tier will be disabled
        gpu = create_gpu_provider();
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
        for (size_t i = 0; i < slots_all.size(); i++) {
            if (slots_pinned[i])
                gpu->free_pinned(slots_all[i]);
            else
                ::free(slots_all[i]);
        }
        delete gpu;
    }


    int classify(const endpoint_t &e) {
        if (e.fd >= 0)
            return K_FILE;
        if (gpu != nullptr && gpu->is_device(e.ptr))
            return K_DEVICE;
        return K_HOST;
    }

    // This method is kept public for deferred initialization of the GPU tier, since we need to know which device to use for multi-GPU systems.
    // It is called from transfer_engine_t::init_tier() when a device pointer is seen for the first time.
    // As we are using Pimpl Idiom pattern this is called inside transfer_engin_t::init_tier and needs to be public
    // If the GPU tier is already initialized or not exist, this method does nothing.
    void init_colocated_gpu_buffer(void *ptr) {
        if (gpu == nullptr)
            return;
        if (classify(mem(ptr)) != K_DEVICE)
            return;
        if (gpu->device_buffer_initialized())
            return;
        int num_gpu_stage_slots = VELOC_GPU_STAGE_SLOTS;
        cfg.get_optional<int>("per-gpu_stage_slots", num_gpu_stage_slots);
        int device_id = gpu->get_device_id(ptr);
        INFO(
            "Initiating device staging pool on GPU " << device_id << ": " << num_gpu_stage_slots <<
            " slots of " << chunk << std::endl
        );
        if (!gpu->init_device_buffer(chunk, num_gpu_stage_slots, device_id))
            ERROR("failed to initialize device staging buffer on GPU " << device_id);
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

    void run_device_out(const std::shared_ptr<xfer_group_impl_t> &g, std::vector<dchunk_t> &chunks) {
        // Three-tier device->file drain. The application's device memory is
        // first snapshotted with a fast D2D into the device staging pool; when
        // that pool is full, the oldest device-staged chunk is evicted (D2H)
        // into a pinned slot and the slot is reused; when the pinned pool is
        // also full, the oldest pinned chunk is written to the file. This keeps
        // the device pool "hot" (incoming chunks keep landing on-device) and
        // guarantees the file is written in capture order, since pin_staged
        // always holds strictly older data than dev_staged.
        std::deque<std::pair<void *, dchunk_t>> dev_staged; // device slots holding app data
        std::deque<std::pair<void *, dchunk_t>> pin_staged; // pinned slots awaiting file write

        // Return every held slot to its pool (used on failure so later groups
        // are not starved; data in these slots is not written to the file).
        auto release_all = [&]() {
            for (auto &p : dev_staged)
                gpu->release_device_slot(p.first);
            dev_staged.clear();
            for (auto &p : pin_staged)
                free_slots.push_back(p.first);
            pin_staged.clear();
        };

        // Evict the oldest device-staged chunk into a free pinned slot. The
        // device slot is released either way; returns false on copy failure.
        auto move_dev_to_pin = [&]() -> bool {
            void *pin = free_slots.back();
            free_slots.pop_back();
            void *dev = dev_staged.front().first;
            if (!gpu->copy_d2h(pin, dev, dev_staged.front().second.n)) {
                fail(g);
                gpu->release_device_slot(dev);
                free_slots.push_back(pin);
                dev_staged.pop_front();
                return false;
            }
            pin_staged.push_back({pin, dev_staged.front().second});
            gpu->release_device_slot(dev);
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
            free_slots.push_back(pin);
            pin_staged.pop_front();
        };

        size_t i = 0;
        while (i < chunks.size() && !g->failed) {
            const dchunk_t &c = chunks[i];
            void *devptr = gpu->acquire_device_slot();
            if (devptr != nullptr) {
                if (!gpu->copy_d2d(devptr, c.dev, c.n)) {
                    fail(g);
                    gpu->release_device_slot(devptr);
                    break;
                }
                add_progress(g, c.n, 0); // application device memory released
                dev_staged.push_back({devptr, c});
                i++;
                continue;
            }
            if (!dev_staged.empty() && !free_slots.empty()) {
                if (!move_dev_to_pin()) // frees a device slot
                    break;
                continue; // stage into the freed device slot
            }
            if (!free_slots.empty()) {
                // No device pool in use (init failed): stage into pinned.
                void *pin = free_slots.back();
                free_slots.pop_back();
                if (!gpu->copy_d2h(pin, c.dev, c.n)) {
                    fail(g);
                    free_slots.push_back(pin);
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
            if (free_slots.empty()) {
                drain_pin_to_file();
            }
        }

        // Flush remaining tiers to the file, oldest-first.
        while (!g->failed) {
            if (!dev_staged.empty()) {
                if (free_slots.empty()) {
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
        std::vector<dchunk_t> dev_out;
        for (auto &t : g->transfers) {
            if (g->failed)
                break;
            int kf = classify(t.from), kt = classify(t.to);
            if (kf == K_HOST && kt == K_FILE) {
                if (!file_provider::write_range(t.to.fd, t.to.off, t.from.ptr, t.len))
                    fail(g);
                else
                    add_progress(g, t.len, t.len);
            } else if (kf == K_DEVICE && kt == K_FILE) {
                for (size_t o = 0; o < t.len; o += chunk) {
                    size_t n = std::min(chunk, t.len - o);
                    dev_out.push_back({(char *)t.from.ptr + o, t.to.fd, t.to.off + o, n});
                }
            } else if (kf == K_FILE && kt == K_HOST) {
                if (!file_provider::read_range(t.from.fd, t.from.off, t.to.ptr, t.len))
                    fail(g);
                else
                    add_progress(g, t.len, t.len);
            } else if (kf == K_FILE && kt == K_DEVICE) {
                for (size_t o = 0; o < t.len && !g->failed; o += chunk) {
                    size_t n = std::min(chunk, t.len - o);
                    void *slot = free_slots.empty() ? nullptr : free_slots.back();
                    if (slot == nullptr) {
                        ERROR("no staging slot available for device restore");
                        fail(g);
                        break;
                    }
                    free_slots.pop_back();
                    if (!file_provider::read_range(t.from.fd, t.from.off + o, slot, n) ||
                        !gpu->copy_h2d((char *)t.to.ptr + o, slot, n))
                        fail(g);
                    else
                        add_progress(g, n, n);
                    free_slots.push_back(slot);
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
        if (!g->failed && !dev_out.empty())
            run_device_out(g, dev_out);
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
    pimpl->init_colocated_gpu_buffer(ptr);
}
