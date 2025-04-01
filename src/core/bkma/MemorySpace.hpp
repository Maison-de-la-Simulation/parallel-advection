#pragma once
#include <sycl/sycl.hpp>

#ifdef SYCL_IMPLEMENTATION_ONEAPI
#define GET_POINTER get_multi_ptr<sycl::access::decorated::no>().get
#else
#define GET_POINTER get_pointer
#endif

//==============================================================================
//==============================================================================
enum class MemorySpace { Local, Global };

template <MemorySpace MemType> struct MemAllocator;

template <MemorySpace MemType>
static inline size_t compute_index(const sycl::nd_item<3> &itm,
                                   unsigned short dim);

// ==========================================
// ==========================================
/* Local memory functions */
template <> struct MemAllocator<MemorySpace::Local> {
    local_acc acc_;
    extents_t extents_;

    [[nodiscard]] MemAllocator(sycl::range<3> range, sycl::handler &cgh)
        : acc_(range, cgh), extents_(range.get(0), range.get(1), range.get(2)) {
    }
    [[nodiscard]] inline auto get_pointer() const { return acc_.GET_POINTER(); }

    [[nodiscard]] inline auto get_extents() const { return extents_; }
};

template <>
inline size_t
compute_index<MemorySpace::Local>(const sycl::nd_item<3> &itm,
                                  unsigned short dim) {
    return itm.get_local_id(dim);
}

// ==========================================
// ==========================================
/* Global memory functions */
template <> struct MemAllocator<MemorySpace::Global> {
    span3d_t data_;

    [[nodiscard]] MemAllocator(span3d_t global_scratch_)
        : data_(global_scratch_){};

    [[nodiscard]] inline size_t compute_index(const sycl::nd_item<3> &itm,
                                              unsigned short dim) {
        return itm.get_global_id(dim);
    }

    [[nodiscard]] inline auto get_pointer() const {
        return data_.data_handle();
    }

    [[nodiscard]] inline auto get_extents() const {
        return extents_t{data_.extent(0), data_.extent(1), data_.extent(2)};
    }
};

template <>
inline size_t
compute_index<MemorySpace::Global>(const sycl::nd_item<3> &itm,
                                   unsigned short dim) {
    return itm.get_global_id(dim);
}
