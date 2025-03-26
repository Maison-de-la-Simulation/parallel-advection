#pragma once
#include <cstdlib>
#include <experimental/mdspan>
#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>

using real_t = double; // tab[C]

using mdspan3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;
using mdspan2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;

// ==========================================
// ==========================================
/* Specifies the number of kernels to run in global/local memory */
struct KernelDispatch {
    size_t k_local_;
    size_t k_global_;
};

// ==========================================
// ==========================================
struct WorkGroupDispatch {
    /* Sequential size */
    size_t s0_ = 1,
           s2_ = 1;   // how many items will one work-item sequentially process

    /* Number of work groups for one batch */
    size_t g0_ = 1, g2_ = 1;   // number of work groups in dim0 and dim2

    // ==========================================
    // ==========================================
    inline void set_num_work_groups(const size_t &n0, const size_t &n2,
                                    const size_t &n_batchs0,
                                    const size_t &n_batchs2, const size_t &w0,
                                    const size_t &w2) {
        if (s0_ > n0 || s2_ > n2) {
            throw std::invalid_argument(
                "s0_ > n0 || s2_ > n2. Sequential size "
                "cannot be larger than dimension size.");
        }
        if (s0_ * w0 > n0 || s2_ * w2 > n2) {
            std::cout << "s0 = " << s0_ << std::endl;
            std::cout << "w0 = " << w0 << std::endl;
            std::cout << "s2 = " << s2_ << std::endl;
            std::cout << "w2 = " << w2 << std::endl;
            throw std::invalid_argument(
                "s0_*w0 > n0 || s2_*w2 > n2. A single work-group cannot "
                "process more than dimension size.");
        }

        g0_ = n0 / n_batchs0 / s0_ / w0;
        g2_ = n2 / n_batchs2 / s2_ / w2;
    }
};

// ==========================================
// ==========================================
struct WorkItemDispatch {
    size_t w0_, w1_, w2_;

    [[nodiscard]]
    inline size_t size() const {
        return w0_ * w1_ * w2_;
    };

    [[nodiscard]]
    sycl::range<3> range() const {
        return sycl::range(w0_, w1_, w2_);
    }

    // ==========================================
    // ==========================================
    inline void set_ideal_sizes(const size_t pref_wg_size, const size_t n0,
                                const size_t n1, const size_t n2) {
        w0_ = 1;
        if (n2 >= pref_wg_size) {
            w1_ = 1;
            w2_ = pref_wg_size;
        } else {
            if (n1 * n2 >= pref_wg_size) {
                w1_ = pref_wg_size / n2;
                w2_ = n2;
            } else {
                // Not enough n1*n2 to fill up work group, we use more from n0
                w0_ = sycl::floor(static_cast<float>(pref_wg_size / n1 * n2));
                w1_ = n1;
                w2_ = n2;
            }
        }
    }   // end set_ideal_sizes

    // ==========================================
    // ==========================================
    inline void adjust_sizes_mem_limit(const size_t &max_elems_alloc,
                                       const size_t &alloc_size) {
        /*TODO: do we need to check if w0*n1>max_mem ?? because if w0 has a lot
        of elements it means that there are fewer elements on n2 and n1 than
        pref_w? so it's not possible to exceed memory in that case right? */
        /* Adjust based on maximum memory available*/
        auto total_wi = size();
        if (w2_ * alloc_size >= max_elems_alloc) {
            w2_ = sycl::floor(static_cast<float>(max_elems_alloc / alloc_size));
            w0_ = 1;
            // Ajuster w1 pour conserver le nombre total de work-items
            w1_ = total_wi / w2_ * w0_;
        }
    }

};   // end struct WorkItemDispatch

// ==========================================
// ==========================================
struct BatchConfig1D {
    size_t n_batch_;
    size_t batch_size_;
    size_t last_batch_size_;

    [[nodiscard]]
    inline size_t offset(const size_t &ibatch) const {
        return batch_size_ * ibatch;
    }
};

[[nodiscard]] inline BatchConfig1D
init_1d_blocking(const size_t n, const size_t max_batchs) noexcept {
    BatchConfig1D bconf;
    /* If there is no problem, the max batch is n */
    bconf.batch_size_ = n < max_batchs ? n : max_batchs;

    /* Compute number of batchs */
    float div = static_cast<float>(n) / static_cast<float>(max_batchs);
    auto floor_div = sycl::floor(static_cast<float>(div));
    auto div_is_int = div == floor_div;
    bconf.n_batch_ = div_is_int ? div : floor_div + 1;

    bconf.last_batch_size_ = div_is_int ? max_batchs : (n % max_batchs);

    return bconf;
}

[[nodiscard]] inline KernelDispatch
dispatch_kernels(const size_t n_kernels, const size_t p) noexcept {
    KernelDispatch kd;
    auto div = n_kernels * p;
    kd.k_local_ = sycl::floor(static_cast<float>(div));
    kd.k_global_ = n_kernels - kd.k_local_;

    return kd;
}   // end dispach_kernels

//==============================================================================
//==============================================================================
//==============================================================================

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
template <MemorySpace MemType, class MySolver>
inline sycl::event
submit_kernels(sycl::queue &Q, mdspan3d_t data, const MySolver &solver,
               const size_t b0_size, const size_t b0_offset,
               const size_t b2_size, const size_t b2_offset,
               const size_t orig_w0, const size_t w1, const size_t orig_w2,
               WorkGroupDispatch wg_dispatch, double *global_scratch = nullptr) {

    const auto w0 = sycl::min(orig_w0, b0_size);
    const auto w2 = sycl::min(orig_w2, b2_size);

    wg_dispatch.set_num_work_groups(b0_size, b2_size, 1, 1, w0, w2);
    auto const seq_size0 = wg_dispatch.s0_;
    auto const seq_size2 = wg_dispatch.s2_;
    auto const g0 = wg_dispatch.g0_;
    auto const g2 = wg_dispatch.g2_;

    const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    const sycl::range<3> local_size(w0, w1, w2);

    auto n0 = data.extent(0);
    auto n1 = data.extent(1);
    auto n2 = data.extent(2);

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
                // mdspan3d_t fdist(data, n0, n1, n2);
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
                            data, global_i0, std::experimental::full_extent,
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

struct BkmaOptimParams {
    BatchConfig1D dispatch_d0;
    BatchConfig1D dispatch_d2;
    size_t w0;
    size_t w1;
    size_t w2;
    WorkGroupDispatch wg_dispatch;
    MemorySpace mem_space;
};

template <class MySolver>
inline sycl::event
bkma_run(sycl::queue &Q, mdspan3d_t data, const MySolver &solver,
         BkmaOptimParams optim_params) {

    sycl::event last_event;

    auto const &n_batch0 = optim_params.dispatch_d0.n_batch_;
    auto const &n_batch2 = optim_params.dispatch_d2.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = optim_params.dispatch_d0.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = optim_params.dispatch_d2.offset(i2_batch);

            auto &batch_size_d0 =
                last_i0 ? optim_params.dispatch_d0.last_batch_size_
                        : optim_params.dispatch_d0.batch_size_;
            auto &batch_size_d2 =
                last_i2 ? optim_params.dispatch_d2.last_batch_size_
                        : optim_params.dispatch_d2.batch_size_;

            last_event.wait();
            switch (optim_params.mem_space) {
            case MemorySpace::Local: {
                last_event = submit_kernels<MemorySpace::Local>(
                    Q, data, solver, batch_size_d0, offset_d0,
                    batch_size_d2, offset_d2, optim_params.w0, optim_params.w1,
                    optim_params.w2, optim_params.wg_dispatch);
            } break;

            case MemorySpace::Global: {
                last_event = submit_kernels<MemorySpace::Global>(
                    Q, data, solver, batch_size_d0, offset_d0,
                    batch_size_d2, offset_d2, optim_params.w0, optim_params.w1,
                    optim_params.w2, optim_params.wg_dispatch);
            } break;

            default: {
                throw std::invalid_argument("Unknown MemorySpace");
            }
            }

        }
    }

    return last_event;
}
