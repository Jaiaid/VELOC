#include "common/tiers/mem_provider.hpp"

#include <algorithm>
#include <cstdlib>
#include <numa.h>
#include <sys/mman.h>

//#define __DEBUG
#include "common/debug.hpp"

// Without a GPU runtime, page-lock with mlock (with CUDA, cuda_provider.cpp
// registers the memory with the runtime instead).
#ifndef WITH_CUDA
bool pin_host_memory(void *ptr, size_t size) { return mlock(ptr, size) == 0; }
void unpin_host_memory(void *ptr, size_t size) { munlock(ptr, size); }
#endif

host_provider_t::host_provider_t(int numa_node, size_t slot_size, int slots) :
    mem_provider_t(mem_dev_t(DEV_CPU, numa_node), slot_size, slots) { }

host_provider_t::~host_provider_t() {
    release_buffer();
}

void *host_provider_t::alloc_slot(size_t size) {
    void *p = numa_available() < 0 ? ::malloc(size) : numa_alloc_onnode(size, get_device_type().id);
    if (p == nullptr)
        return nullptr;
    if (pin_host_memory(p, size))
        pinned_.push_back(p);
    else
        ERROR("cannot pin host staging slot on NUMA node " << get_device_type().id << ", using pageable memory");
    return p;
}

void host_provider_t::free_slot(void *ptr) {
    auto it = std::find(pinned_.begin(), pinned_.end(), ptr);
    if (it != pinned_.end()) {
        unpin_host_memory(ptr, slot_size());
        pinned_.erase(it);
    }
    if (numa_available() < 0)
        ::free(ptr);
    else
        numa_free(ptr, slot_size());
}
