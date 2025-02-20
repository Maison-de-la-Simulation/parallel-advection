#pragma once

#include <Solver.h>
#include <cstddef>
#include <sycl/sycl.hpp>

// ==========================================
// ==========================================
class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver) = 0;
};

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
                w0_ = std::floor(pref_wg_size / n1 * n2);
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
            w2_ = std::floor(max_elems_alloc / alloc_size);
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

    // [[nodiscard]]
    // inline size_t start_index(const size_t &ibatch) const {
    //     return offset(ibatch);
    // }

    // [[nodiscard]]
    // inline size_t stop_index(const size_t &ibatch) const {
    //     return ibatch == n_batch_ - 1 ? offset(ibatch) + last_batch_size_
    //                                   : offset(ibatch) + batch_size_;
    // }
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

// ==========================================
// ==========================================
inline void
print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
    if (lvl == 0)
        std::cout << "--------------------------------" << std::endl;
    std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
              << r.get(2) << "}" << std::endl;
}

// ==========================================
// ==========================================
[[nodiscard]] inline KernelDispatch
init_kernel_splitting(const float p, const size_t n) {
    KernelDispatch k_dispatch;

    auto div =  n * p;
    k_dispatch.k_local_ = std::floor(div);

    k_dispatch.k_global_ =  n - k_dispatch.k_local_;

    return k_dispatch;
}

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
