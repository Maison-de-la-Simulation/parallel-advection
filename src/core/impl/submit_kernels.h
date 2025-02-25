#include <sycl/sycl.hpp>

enum class MemoryType { Local, Global };

template <MemoryType MemType>
struct MemoryAllocator;

// Specialization for Local Memory
template <>
struct MemoryAllocator<MemoryType::Local> {
    [[nodiscard]] static inline auto allocate(sycl::handler& cgh, size_t size) {
        return sycl::accessor<int, 1, sycl::access::mode::read_write,
                              sycl::access::target::local>(sycl::range<1>(size), cgh);
    }

    [[nodiscard]] static inline size_t compute_index(const sycl::nd_item<1>& item) {
        return item.get_local_id(0); // Local memory indexing
    }
};

// Specialization for Global Memory
template <>
struct MemoryAllocator<MemoryType::Global> {
    [[nodiscard]] static inline auto allocate(sycl::handler& cgh, sycl::buffer<int, 1>& buf) {
        return buf.get_access<sycl::access::mode::read_write>(cgh);
    }

    [[nodiscard]] static inline size_t compute_index(const sycl::nd_item<1>& item) {
        return item.get_global_id(0); // Global memory indexing
    }
};

// ** Templated Function Using Memory Policy **
template <MemoryType MemType>
void process_data(sycl::queue& q, size_t N) {
    sycl::buffer<int, 1> global_buf(N);  // Only needed for global memory

    q.submit([&](sycl::handler& cgh) {
        auto data = MemoryAllocator<MemType>::allocate(cgh, global_buf);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(64)), 
                         [=](sycl::nd_item<1> item) {
            const size_t index = MemoryAllocator<MemType>::compute_index(item);
            data[index] = index; // Example operation
        });
    });
}
