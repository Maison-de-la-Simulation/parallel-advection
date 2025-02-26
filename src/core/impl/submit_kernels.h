#pragma once
#include "IAdvectorX.h"
#include "advectors.h"
#include <experimental/mdspan>

enum class MemorySpace { Local, Global };

template <MemorySpace MemType> struct MemAllocator;

using local_acc = sycl::local_accessor<double, 3>;
using extents_t =
    std::experimental::extents<std::size_t, std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent>;

template <MemorySpace MemType>
static inline size_t compute_index(const sycl::nd_item<3> &itm, int dim);

// ==========================================
// ==========================================
/* Local memory functions */
template <> struct MemAllocator<MemorySpace::Local> {
    local_acc acc_;
    extents_t extents_;

    [[nodiscard]] MemAllocator(sycl::range<3> range, sycl::handler &cgh)
        : acc_(range, cgh), extents_(range.get(0), range.get(1), range.get(2)) {
    }
    [[nodiscard]] inline auto get_pointer() const { return acc_.get_pointer(); }
};

template <>
inline size_t
compute_index<MemorySpace::Local>(const sycl::nd_item<3> &itm, int dim) {
    return itm.get_local_id(dim);
}

// ==========================================
// ==========================================
/* Global memory functions */
template <> struct MemAllocator<MemorySpace::Global> {
    double *ptr_;
    extents_t extents_;

    [[nodiscard]] MemAllocator(double *ptr, extents_t extents)
        : ptr_(ptr), extents_(extents){};

    [[nodiscard]] inline size_t compute_index(const sycl::nd_item<3> &itm,
                                              int dim) {
        return itm.get_global_id(dim);
    }

    [[nodiscard]] inline auto get_pointer() const { return ptr_; }
};
template <>
inline size_t
compute_index<MemorySpace::Global>(const sycl::nd_item<3> &itm, int dim) {
    return itm.get_global_id(dim);
}

// ==========================================
// ==========================================
template <MemorySpace MemType>
inline sycl::event
submit_kernels(sycl::queue &Q, double *fdist_dev, const Solver &solver,
               const size_t b0_size, const size_t b0_offset,
               const size_t b2_size, const size_t b2_offset,
               const size_t orig_w0, const size_t w1, const size_t orig_w2,
               WorkGroupDispatch wg_dispatch, const size_t n0, const size_t n1,
               const size_t n2, double *global_scratch = nullptr) {

    const auto w0 = sycl::min(orig_w0, b0_size);
    const auto w2 = sycl::min(orig_w2, b2_size);

    wg_dispatch.set_num_work_groups(b0_size, b2_size, 1, 1, w0, w2);
    auto const seq_size0 = wg_dispatch.s0_;
    auto const seq_size2 = wg_dispatch.s2_;
    auto const g0 = wg_dispatch.g0_;
    auto const g2 = wg_dispatch.g2_;

    const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    const sycl::range<3> local_size(w0, w1, w2);

    // print_range("global_size", global_size);
    // print_range("local_size", local_size, 1);
    // /* Debug */
    // if constexpr (MemType == MemorySpace::Local) {
    //     std::cout << "in submit local kernel" << std::endl;
    // } else {
    //     std::cout << "in submit global kernel" << std::endl;
    // }

    return Q.submit([&](sycl::handler &cgh) {
        auto mallocator = [&]() {
            if constexpr (MemType == MemorySpace::Local) {
                sycl::range<3> acc_range(w0, w2, n1);
                return MemAllocator<MemType>(acc_range, cgh);
            } else {
                extents_t ext(b0_size, n2, n1);
                return MemAllocator<MemType>(global_scratch, ext);
            }
        }();

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(mallocator.get_pointer(), mallocator.extents_);

                const auto i1 = itm.get_local_id(1);
                const auto local_i0 = compute_index<MemType>(itm, 0);
                const auto local_i2 = compute_index<MemType>(itm, 2);

                auto scratch_slice = std::experimental::submdspan(
                    scr, local_i0, local_i2, std::experimental::full_extent);

                const auto start_idx0 = b0_offset + itm.get_global_id(0);
                const auto stop_idx0 = sycl::min(n0, start_idx0 + b0_size);
                for (size_t global_i0 = start_idx0; global_i0 < stop_idx0;
                     global_i0 += g0 * w0) {

                    const auto start_idx2 = b2_offset + itm.get_global_id(2);
                    const auto stop_idx2 = sycl::min(n2, start_idx2 + b2_size);
                    for (size_t global_i2 = start_idx2; global_i2 < stop_idx2;
                         global_i2 += g2 * w2) {

                        auto data_slice = std::experimental::submdspan(
                            fdist, global_i0, std::experimental::full_extent,
                            global_i2);

                        for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                            scratch_slice(ii1) =
                                solver(data_slice, global_i0, ii1, global_i2);
                        }

                        sycl::group_barrier(itm.get_group());

                        for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                            data_slice(ii1) = scratch_slice(ii1);
                        }
                    }   // end for ii2
                }   // end for ii0
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}

// Wrappers
inline sycl::event
submit_local_kernels(sycl::queue &Q, double *fdist_dev, const Solver &solver,
                     const size_t b0_size, const size_t b0_offset,
                     const size_t b2_size, const size_t b2_offset,
                     const size_t orig_w0, const size_t w1,
                     const size_t orig_w2, WorkGroupDispatch wg_dispatch,
                     const size_t n0, const size_t n1, const size_t n2) {
    return submit_kernels<MemorySpace::Local>(
        Q, fdist_dev, solver, b0_size, b0_offset, b2_size, b2_offset, orig_w0,
        w1, orig_w2, wg_dispatch, n0, n1, n2);
}

inline sycl::event
submit_global_kernels(sycl::queue &Q, double *fdist_dev, const Solver &solver,
                      const size_t b0_size, const size_t b0_offset,
                      const size_t b2_size, const size_t b2_offset,
                      const size_t orig_w0, const size_t w1,
                      const size_t orig_w2, WorkGroupDispatch wg_dispatch,
                      const size_t n0, const size_t n1, const size_t n2,
                      double *global_scratch) {
    return submit_kernels<MemorySpace::Global>(
        Q, fdist_dev, solver, b0_size, b0_offset, b2_size, b2_offset, orig_w0,
        w1, orig_w2, wg_dispatch, n0, n1, n2, global_scratch);
}
