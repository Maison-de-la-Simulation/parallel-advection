#pragma once
#include <sycl/sycl.hpp>
#include <experimental/mdspan>

struct ConvSolver {
    // ==========================================
    // ==========================================
    virtual auto inline window() const = 0;

    // ==========================================
    // ==========================================
    template <class ArrayLike1D>
    inline __attribute__((always_inline))
    real_t operator()(const ArrayLike1D scr,
                      const size_t &,
                      const size_t &i1,
                      const size_t &) const = 0;
};
