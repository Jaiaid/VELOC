#include "common/tiers/mem_provider.hpp"

// ---- mem_dev_t -------------------------------------------------------------

mem_dev_t::mem_dev_t(int t, int i) : type(t), id(i) { }

bool mem_dev_t::operator==(const mem_dev_t &o) const {
    return type == o.type && id == o.id;
}

// ---- mem_provider_t --------------------------------------------------------

mem_provider_t::mem_provider_t(mem_dev_t d, size_t slot_size, int slots) :
    dev(d), slot_size_(slot_size), slot_count_(slots) { }

mem_provider_t::~mem_provider_t() { }

void mem_provider_t::release_buffer() {
    std::lock_guard<std::mutex> lk(mtx_);
    for (void *p : slots_all_)
        free_slot(p);
    slots_all_.clear();
    slots_free_.clear();
}

mem_dev_t mem_provider_t::get_device_type() const {
    return dev;
}

size_t mem_provider_t::slot_size() const {
    return slot_size_;
}

bool mem_provider_t::init_buffer() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!slots_all_.empty())
        return true;
    if (slot_count_ < 1 || slot_size_ == 0)
        return false;
    for (int i = 0; i < slot_count_; i++) {
        void *p = alloc_slot(slot_size_);
        if (p == nullptr)
            break;
        slots_all_.push_back(p);
        slots_free_.push_back(p);
    }
    return !slots_all_.empty();
}

bool mem_provider_t::buffer_initialized() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return !slots_all_.empty();
}

void *mem_provider_t::acquire_slot() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (slots_free_.empty())
        return nullptr;
    void *p = slots_free_.back();
    slots_free_.pop_back();
    return p;
}

void mem_provider_t::release_slot(void *ptr) {
    if (ptr == nullptr)
        return;
    std::lock_guard<std::mutex> lk(mtx_);
    slots_free_.push_back(ptr);
}

size_t mem_provider_t::free_slots() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return slots_free_.size();
}

// ---- gpu_provider_t --------------------------------------------------------

gpu_provider_t::gpu_provider_t(int device_id, int numa_node, size_t slot_size, int slots) :
    mem_provider_t(mem_dev_t(DEV_GPU, device_id), slot_size, slots), numa_node_(numa_node) { }

gpu_provider_t::~gpu_provider_t() { }

int gpu_provider_t::affiliated_numa_node() const {
    return numa_node_;
}
