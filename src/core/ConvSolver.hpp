#pragma once

#include <sycl/sycl.hpp>
#include <experimental/mdspan>
#include "../types.hpp"

struct ConvSolver {
    span3d_t weight_span_;
    span1d_t bias_span_;
    size_t kernel_size_;
    size_t in_channels_;
    size_t input_length_;
    static constexpr size_t stride_  = 1;
    static constexpr size_t padding_ = 0;

    auto inline window() const {return kernel_size_;}
    // ==========================================
    // ==========================================
    /* The _solve_ function of the algorithm presented */
    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D scr,
                      const size_t &,
                      const size_t &i1,
                      const size_t &) const {
        size_t out_channels = in_channels_; //constraint

        auto oc = i1/input_length_;
        auto i_l = i1 - oc*input_length_;

        real_t sum = bias_span_(oc);
        for (int ic = 0; ic < in_channels_; ++ic) {
            for (int k = 0; k < kernel_size_; ++k) {
                // int input_idx = i1 * stride_ + k - padding_;
                int input_idx = i_l - k;

                if (input_idx < scr.extent(0)) {
                    sum += scr(input_idx) * weight_span_(k, ic, oc);
                }
            }
        }

        return sum;
    }
};
