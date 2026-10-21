#ifndef __MEM_PROVIDER_HPP
#define __MEM_PROVIDER_HPP

#include <cstddef>
#include <mutex>
#include <vector>

enum { DEV_CPU = 0, DEV_GPU = 1 };

// Location of a memory tier. Not named dev_t to avoid clashing with POSIX
// dev_t from <sys/types.h>.
struct mem_dev_t {
    int type; // DEV_CPU or DEV_GPU
    int id;   // device id, for CPU id indicates NUMA ID, for GPU id indicates GPU ID
    mem_dev_t(int t, int i);
    bool operator==(const mem_dev_t &o) const;
};

// One memory tier (a NUMA node's host memory or a single GPU) owning a pool of
// `slots` staging buffers of `slot_size` bytes. The provider is constructed
// eagerly for every tier present on the node, but the pool is allocated lazily
// by init_buffer(), i.e. only once a protected pointer needs this tier.
class mem_provider_t {
    mem_dev_t dev;
    size_t slot_size_;
    int slot_count_;
    std::vector<void *> slots_all_;  // every allocated slot (for teardown)
    std::vector<void *> slots_free_;
    mutable std::mutex mtx_;

protected:
    // Allocate/free one staging slot in this tier's memory.
    virtual void *alloc_slot(size_t size) = 0;
    virtual void free_slot(void *ptr) = 0;

    // Derived destructors must call this, since free_slot() is virtual.
    void release_buffer();

public:
    mem_provider_t(mem_dev_t d, size_t slot_size, int slots);

    // Tier this provider manages, fixed at construction.
    mem_dev_t get_device_type() const;
    size_t slot_size() const;

    // Allocate the slot pool (no-op if already done). Returns false if not even
    // a single slot could be allocated.
    bool init_buffer();
    bool buffer_initialized() const;
    void *acquire_slot(); // nullptr when all slots busy
    void release_slot(void *ptr);
    size_t free_slots() const;

    virtual ~mem_provider_t();
};

// Host memory of one NUMA node. Slots are allocated on that node with libnuma
// and page-locked via pin_host_memory() (falls back to unpinned on failure).
class host_provider_t : public mem_provider_t {
    std::vector<void *> pinned_; // slots successfully page-locked

protected:
    void *alloc_slot(size_t size) override;
    void free_slot(void *ptr) override;

public:
    host_provider_t(int numa_node, size_t slot_size, int slots);
    ~host_provider_t() override;
};

// Vendor-neutral abstraction over a single GPU. All CUDA (or other vendor)
// headers are confined to the corresponding provider translation unit (e.g.
// cuda_provider.cpp), so the rest of the code never sees them.
class gpu_provider_t : public mem_provider_t {
    int numa_node_; // NUMA node the GPU is attached to, fixed at construction

public:
    gpu_provider_t(int device_id, int numa_node, size_t slot_size, int slots);

    // Blocking device<->host copies (executed on the engine progress thread, so
    // they never block the application thread).
    virtual bool copy_d2h(void *host_dst, const void *dev_src, size_t size) = 0;
    virtual bool copy_h2d(void *dev_dst, const void *host_src, size_t size) = 0;

    // Blocking device<->device copy within the same GPU.
    virtual bool copy_d2d(void *dst, const void *src, size_t size) = 0;

    // NUMA node the GPU is attached to (its host staging tier), -1 if unknown.
    int affiliated_numa_node() const;

    virtual ~gpu_provider_t();
};

// GPU runtime entry points, implemented by the vendor provider translation
// unit. Without GPU support they fall back to: no devices, no device pointers,
// nullptr provider, and mlock() for pinning.
int gpu_device_count();
// GPU id owning ptr, or -1 if ptr is not plain device memory.
int gpu_device_of(const void *ptr);
// Returns a provider for GPU device_id, or nullptr if unavailable.
gpu_provider_t *create_gpu_provider(int device_id, size_t slot_size, int slots);
// Page-lock host memory so it can be used for DMA by the GPU runtime.
bool pin_host_memory(void *ptr, size_t size);
void unpin_host_memory(void *ptr, size_t size);

#endif // __MEM_PROVIDER_HPP
