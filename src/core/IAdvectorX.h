#pragma once

#include <Solver.h>
#include <cstddef>
#include <sycl/sycl.hpp>

class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver) = 0;
};

/* Specifies the number of kernels to run in global/local memory */
struct KernelDispatch {
    size_t k_local_;
    size_t k_global_;
};

struct WorkGroupDispatch {
    /* Sequential size */
    size_t s0_ = 1, s2_ = 1; // how many items will one work-item sequentially process

    /* Number of work groups for one batch */
    size_t g0_ = 1, g2_ = 1; // number of work groups in dim0 and dim2

    inline void set_num_work_groups(const size_t &n0, const size_t &n2,
                                    const size_t &n_batchs0,
                                    const size_t &n_batchs2, const size_t &w0,
                                    const size_t &w2) {
        g0_ = std::ceil(n0 / n_batchs0 / s0_ / w0);
        g2_ = std::ceil(n2 / n_batchs2 / s2_ / w2);
    }
};

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

    // =========================================================================
    // =========================================================================
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
                w0_ = std::floor(pref_wg_size / n1 * n2);
                w1_ = n1;
                w2_ = n2;
            }
        }
    }   // end set_ideal_sizes

    // =========================================================================
    // =========================================================================
    inline void adjust_sizes_mem_limit(const size_t &max_elems_alloc,
                                       const size_t &alloc_size) {
        /*TODO: do we need to check if w0*n1>max_mem ?? because if w0 has a lot
of elements it means that there are fewer elements on n2 and n1 than pref_w? so
it's not possible to exceed memory in that case right? */
        /* Adjust based on maximum memory available*/
        if (w2_ * alloc_size >= max_elems_alloc) {
            w2_ = std::floor(max_elems_alloc / alloc_size);
            w1_ = std::floor(size() / w2_);
            w0_ = 1;
        }
    }

};   // end struct WorkItemDispatch

struct BatchConfig1D {
    size_t n_batch_;
    size_t batch_size_;
    size_t last_batch_size_;

    [[nodiscard]]
    inline size_t offset(const size_t &ibatch) const {
        return batch_size_ * ibatch;
    }

    [[nodiscard]]
    inline size_t start_index(const size_t &ibatch) const {
        return offset(ibatch);
    }

    [[nodiscard]]
    inline size_t stop_index(const size_t &ibatch) const {
        return ibatch == n_batch_ - 1 ? offset(ibatch) + last_batch_size_
                                      : offset(ibatch) + batch_size_;
    }
};

[[nodiscard]] inline BatchConfig1D
init_1d_blocking(const size_t n, const size_t max_batchs) noexcept {
    BatchConfig1D bconf;
    /* If there is no problem, the max batch is n */
    bconf.batch_size_ = n < max_batchs ? n : max_batchs;

    /* Compute number of batchs */
    float div = static_cast<float>(n) / static_cast<float>(max_batchs);
    auto floor_div = std::floor(div);
    auto div_is_int = div == floor_div;
    bconf.n_batch_ = div_is_int ? div : floor_div + 1;

    bconf.last_batch_size_ = div_is_int ? max_batchs : (n % max_batchs);

    return bconf;
}

[[nodiscard]] inline KernelDispatch
dispatch_kernels(const size_t n_kernels, const size_t p) noexcept {
    KernelDispatch kd;
    auto div = n_kernels * p;
    kd.k_local_ = std::floor(div);
    kd.k_global_ = n_kernels - kd.k_local_;

    return kd;
}   // end dispach_kernels

// [[nodiscard]] inline WorkItemDispatch
// compute_ideal_wg_size(const size_t pref_wg_size, const size_t n0,
//                       const size_t n1, const size_t n2) noexcept {
//     WorkItemDispatch dispatch;

//     auto &w0 = dispatch.w0_;
//     auto &w1 = dispatch.w1_;
//     auto &w2 = dispatch.w2_;

//     w0 = 1;
//     if (n2 >= pref_wg_size) {
//         w1 = 1;
//         w2 = pref_wg_size;
//     } else {
//         if (n1 * n2 >= pref_wg_size) {
//             w1 = pref_wg_size / n2;
//             w2 = n2;
//         } else {
//             // Not enough n1*n2 to fill up work group, we use more from n0
//             w0 = std::floor(pref_wg_size / n1 * n2);
//             w1 = n1;
//             w2 = n2;
//         }
//     }

//     return dispatch;
// }   // set_wg_size

inline void
print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
    if (lvl == 0)
        std::cout << "--------------------------------" << std::endl;
    std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
              << r.get(2) << "}" << std::endl;
}

// [[nodiscard]] inline WgDispatch
// adjust_wg_dispatch(const WgDispatch &ideal_wg,
//                    const BlockingDispatch1D &block_d0,
//                    const BlockingDispatch1D &block_d2, const size_t
//                    alloc_size, const size_t max_elem_mem) {
//     WgDispatch adjusted_wg = ideal_wg;

//     /* Adjust based on sizes and divisible ranges */
//     auto const &global_size0 = block_d0.batch_size_;
//     auto const &global_size2 = block_d2.batch_size_;

//     size_t new_w0 = std::min(adjusted_wg.w0_, block_d0.batch_size_);
//     // while (global_size0 % new_w0 != 0) {
//     //     new_w0 -= 1;
//     // }

//     size_t new_w2 = std::min(adjusted_wg.w2_, block_d2.batch_size_);
//     // while (global_size2 % new_w2 != 0) {
//     //     new_w2 -= 1;
//     // }

//     // Ajuster w1 pour conserver le nombre total de work-items
//     total_wg_items = adjusted_wg.size();   // might have changed
//     size_t new_w1 = total_wg_items / (new_w0 * new_w2);

//     adjusted_wg.w0_ = new_w0;
//     adjusted_wg.w1_ = new_w1;
//     adjusted_wg.w2_ = new_w2;

//     print_range("global_size", sycl::range<3>(global_size0, 1,
//     global_size2)); print_range("ideal_wg",
//                 sycl::range<3>(ideal_wg.w0_, ideal_wg.w1_, ideal_wg.w2_), 1);
//     print_range("adjusted_wg", sycl::range<3>(new_w0, new_w1, new_w2), 1);

//     return adjusted_wg;
// }

// [[nodiscard]] inline AdaptiveWgDispatch
// compute_adaptive_wg_dispatch(const WgDispatch &preferred_wg,
//                              const BlockingConfig1D &config_d0,
//                              const BlockingConfig1D &config_d2,
//                              const size_t alloc_size,
//                              const size_t max_elem_mem) {

//     AdaptiveWgDispatch adaptive;

//     // Compute work-group sizes for normal batches
//     adaptive.normal_dispatch_ = adjust_wg_dispatch(
//         preferred_wg, {config_d0.max_batch_size_, 0},
//         {config_d2.max_batch_size_, 0}, alloc_size, max_elem_mem);

//     // Compute for last batch in d0
//     adaptive.last_dispatch_d0_ = adjust_wg_dispatch(
//         preferred_wg, config_d0.last_dispatch_, {config_d2.max_batch_size_,
//         0}, alloc_size, max_elem_mem);

//     // Compute for last batch in d2
//     adaptive.last_dispatch_d2_ =
//         adjust_wg_dispatch(preferred_wg, {config_d0.max_batch_size_, 0},
//                            config_d2.last_dispatch_, alloc_size,
//                            max_elem_mem);

//     // Compute for last batch in both d0 and d2
//     adaptive.last_dispatch_d0_d2_ =
//         adjust_wg_dispatch(preferred_wg, config_d0.last_dispatch_,
//                            config_d2.last_dispatch_, alloc_size,
//                            max_elem_mem);

//     return adaptive;
// }
