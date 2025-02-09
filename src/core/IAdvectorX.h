#pragma once

#include <array>
#include <cstddef>
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

struct WgDispatch{
  size_t w0_;
  size_t w1_;
  size_t w2_;
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
set_wg_size(const size_t pref_wg_size, const size_t max_elem_mem,
            const size_t n0, const size_t n1, const size_t n2) noexcept {
    WgDispatch dispatch;

    dispatch.w0_ = 1;
    if (n2 >= pref_wg_size) {
        dispatch.w1_ = 1;
        dispatch.w2_ = pref_wg_size;
    } else {
        if (n1 * n2 >= pref_wg_size) {
            dispatch.w1_ = pref_wg_size / n2;
            dispatch.w2_ = n2;
        } else {
            // Not enough n1*n2 to fill up work group, we use more from n0
            dispatch.w0_ = std::floor(pref_wg_size / n1 * n2);
            dispatch.w1_ = n1;
            dispatch.w2_ = n2;
        }
    }

  /*TODO: do we need to check if w0*n1>max_mem ?? because if w0 has a lot of elements
  it means that there are fewer elements on n2 and n1 than pref_w? so it's not possible to 
  exceed memory in that case right? */
    if (dispatch.w2_ * n1 >= max_elem_mem) {
        dispatch.w2_ = std::floor(max_elem_mem / n1);
        dispatch.w1_ = std::floor(pref_wg_size / dispatch.w2_);
        dispatch.w0_ = 1;
    }

    return dispatch;
}// set_wg_size

/*TODO: check_work_group_compatibility(WgDispatch, BlockingDispatch){}
If I cannot make a global range with this size on this dimension, returns
a valid wg configuration for this batch size
*/