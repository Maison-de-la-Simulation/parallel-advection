#pragma once

#include <array>
#include <cstddef>
#include <hipSYCL/sycl/libkernel/range.hpp>
#include <sycl/sycl.hpp>
#include <Solver.h>

class IAdvectorX {
  public:
    virtual ~IAdvectorX() = default;

    virtual sycl::event operator()(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver) = 0;

};

/* Specifies the number of kernels to run in global/local memory */
struct KernelDispatch{
    size_t k_local_;
    size_t k_global_;
};

// using WgDispatch = sycl::range<3>;
struct WgDispatch{
  size_t w0_;
  size_t w1_;
  size_t w2_;

  sycl::range<3> range() { return sycl::range<3>{w0_, w1_, w2_}; }
};

struct AdaptiveWgDispatch {
    WgDispatch normal_dispatch;   // Work-group sizes for normal batches
    WgDispatch last_dispatch_d0;  // Work-group sizes when last batch in d0
    WgDispatch last_dispatch_d2;  // Work-group sizes when last batch in d2
    WgDispatch last_dispatch_d0_d2; // Work-group sizes when last batch in both d0 and d2
};


struct BlockingDispatch1D{
  size_t batch_size_;
  size_t offset_;

  void set_offset(size_t & i){offset_ = batch_size_*i;}
};

struct BlockingConfig1D{
  size_t n_batch_;
  size_t max_batch_size_;
  BlockingDispatch1D last_dispatch_;
};

[[nodiscard]] inline BlockingConfig1D
init_1d_blocking(const size_t n, const size_t max_batchs) noexcept {
    BlockingConfig1D bconf;
    bconf.max_batch_size_ = max_batchs;

    /* Compute number of batchs */
    float div =
        static_cast<float>(n) / static_cast<float>(max_batchs);
    auto floor_div = std::floor(div);
    auto div_is_int = div == floor_div;
    bconf.n_batch_ = div_is_int ? div : floor_div + 1;

    bconf.last_dispatch_.batch_size_ =
        div_is_int ? max_batchs : (n % max_batchs);
    bconf.last_dispatch_.offset_ = max_batchs * (bconf.n_batch_ - 1);

    return bconf;
}

[[nodiscard]] inline KernelDispatch
dispatch_kernels(const size_t n_kernels, const size_t p) noexcept {
  KernelDispatch kd;
  auto div = n_kernels * p;
  kd.k_local_ = std::floor(div);
  kd.k_global_ = n_kernels - kd.k_local_;

  return kd;
} // end dispach_kernels

[[nodiscard]] inline WgDispatch
compute_ideal_wg_size(const size_t pref_wg_size, const size_t max_elem_mem,
                      const size_t n0, const size_t n1,
                      const size_t n2) noexcept {
    WgDispatch dispatch;

    auto& w0 = dispatch.w0_;
    auto& w1 = dispatch.w1_;
    auto& w2 = dispatch.w2_;

    w0 = 1;
    if (n2 >= pref_wg_size) {
        w1 = 1;
        w2 = pref_wg_size;
    } else {
        if (n1 * n2 >= pref_wg_size) {
            w1 = pref_wg_size / n2;
            w2 = n2;
        } else {
            // Not enough n1*n2 to fill up work group, we use more from n0
            w0 = std::floor(pref_wg_size / n1 * n2);
            w1 = n1;
            w2 = n2;
        }
    }

  /*TODO: do we need to check if w0*n1>max_mem ?? because if w0 has a lot of elements
  it means that there are fewer elements on n2 and n1 than pref_w? so it's not possible to 
  exceed memory in that case right? */
    if (w2 * n1 >= max_elem_mem) {
        w2 = std::floor(max_elem_mem / n1);
        w1 = std::floor(pref_wg_size / w2);
        w0 = 1;
    }

    return dispatch;
}   // set_wg_size

/*TODO: check_work_group_compatibility(WgDispatch, BlockingDispatch){}
If I cannot make a global range with this size on this dimension, returns
a valid wg configuration for this batch size
*/
// bool check_wg_compat(, const BlockingDispatch1D)
[[nodiscard]] inline WgDispatch
adjust_wg_dispatch(const WgDispatch &ideal_wg,
                   const BlockingDispatch1D &block_d0,
                   const BlockingDispatch1D &block_d2) {
    WgDispatch adjusted_wg = ideal_wg;

    // Adjust work-group sizes to match the batch sizes where necessary
    adjusted_wg.w0_ = std::min(ideal_wg.w0_, block_d0.batch_size_);
    adjusted_wg.w2_ = std::min(ideal_wg.w2_, block_d2.batch_size_);

    return adjusted_wg;
}

[[nodiscard]] inline AdaptiveWgDispatch
compute_adaptive_wg_dispatch(const WgDispatch &preferred_wg,
                             const BlockingConfig1D &config_d0,
                             const BlockingConfig1D &config_d2) {

    AdaptiveWgDispatch adaptive;

    // Compute work-group sizes for normal batches
    adaptive.normal_dispatch = adjust_wg_dispatch(preferred_wg, 
                                                  {config_d0.max_batch_size_, 0}, 
                                                  {config_d2.max_batch_size_, 0});

    // Compute for last batch in d0
    adaptive.last_dispatch_d0 = adjust_wg_dispatch(preferred_wg, 
                                                   config_d0.last_dispatch_, 
                                                   {config_d2.max_batch_size_, 0});

    // Compute for last batch in d2
    adaptive.last_dispatch_d2 = adjust_wg_dispatch(preferred_wg, 
                                                   {config_d0.max_batch_size_, 0}, 
                                                   config_d2.last_dispatch_);

    // Compute for last batch in both d0 and d2
    adaptive.last_dispatch_d0_d2 = adjust_wg_dispatch(preferred_wg, 
                                                      config_d0.last_dispatch_, 
                                                      config_d2.last_dispatch_);

    return adaptive;
}
