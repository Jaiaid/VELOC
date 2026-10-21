#include "common/tiers/mem_provider.hpp"

#include <cuda_runtime.h>

#if CUDART_VERSION < 12020
#include <cctype>
#include <fstream>
#include <string>
#endif

//#define __DEBUG
#include "common/debug.hpp"

// CUDA implementation of the GPU provider. This translation unit is the only
// place in the whole project that includes CUDA headers; it is compiled only
// when WITH_CUDA is defined (see CMake).
namespace {

// Runs a scope with `id` as the calling thread's current device and restores
// the previous one, so the application's CUDA context is left untouched.
struct device_guard_t {
    int prev = -1;
    explicit device_guard_t(int id) {
        if (cudaGetDevice(&prev) != cudaSuccess)
            prev = -1;
        if (prev != id)
            cudaSetDevice(id);
    }
    ~device_guard_t() {
        if (prev >= 0)
            cudaSetDevice(prev);
    }
};

// Host NUMA node closest to the GPU; -1 if unknown. CUDA >= 12.2 exposes it as
// a device attribute; older runtimes read the GPU's PCI device from sysfs.
#if CUDART_VERSION >= 12020
int numa_node_of_gpu(int device_id) {
    int node = -1;
    if (cudaDeviceGetAttribute(&node, cudaDevAttrHostNumaId, device_id) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return node;
}
#else
int numa_node_of_gpu(int device_id) {
    char bus_id[32];
    if (cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), device_id) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    std::string path(bus_id);
    for (auto &c : path)
        c = std::tolower(c);
    std::ifstream f("/sys/bus/pci/devices/" + path + "/numa_node");
    int node = -1;
    if (!(f >> node))
        return -1;
    return node;
}
#endif

class cuda_provider_t : public gpu_provider_t {
    inline int gpu_id() const { return get_device_type().id; }

protected:
    void *alloc_slot(size_t size) override {
        device_guard_t dg(gpu_id());
        void *p = nullptr;
        if (cudaMalloc(&p, size) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        return p;
    }

    void free_slot(void *ptr) override {
        cudaFree(ptr);
    }

public:
    cuda_provider_t(int device_id, size_t slot_size, int slots) :
        gpu_provider_t(device_id, numa_node_of_gpu(device_id), slot_size, slots) { }

    ~cuda_provider_t() override {
        release_buffer();
    }

    // Copies run on the engine progress thread, so switching its current device is harmless.
    bool copy_d2h(void *host_dst, const void *dev_src, size_t size) override {
        cudaSetDevice(gpu_id());
        return cudaMemcpy(host_dst, dev_src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    bool copy_h2d(void *dev_dst, const void *host_src, size_t size) override {
        cudaSetDevice(gpu_id());
        return cudaMemcpy(dev_dst, host_src, size, cudaMemcpyHostToDevice) == cudaSuccess;
    }

    bool copy_d2d(void *dst, const void *src, size_t size) override {
        cudaSetDevice(gpu_id());
        return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice) == cudaSuccess;
    }
};

} // anonymous namespace

int gpu_device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count;
}

int gpu_device_of(const void *ptr) {
    if (ptr == nullptr)
        return -1;
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) != cudaSuccess) {
        // Unregistered host allocations report an error on older runtimes;
        // clear it and treat the pointer as host memory.
        cudaGetLastError();
        return -1;
    }
    // Managed (unified) memory is host-accessible, so we treat only plain
    // device memory as requiring a device-to-host staging copy.
    return attr.type == cudaMemoryTypeDevice ? attr.device : -1;
}

gpu_provider_t *create_gpu_provider(int device_id, size_t slot_size, int slots) {
    return new cuda_provider_t(device_id, slot_size, slots);
}

bool pin_host_memory(void *ptr, size_t size) {
    // Portable: pinned for every CUDA context, not just the current device's.
    if (cudaHostRegister(ptr, size, cudaHostRegisterPortable) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

void unpin_host_memory(void *ptr, size_t) {
    cudaHostUnregister(ptr);
}
