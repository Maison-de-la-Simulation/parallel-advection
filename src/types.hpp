#pragma once
#include <experimental/mdspan>
#include <sycl/sycl.hpp>

using real_t = double;

[[nodiscard]] inline auto
sycl_alloc(size_t size, sycl::queue &q) {
    return sycl::malloc_device<real_t>(size, q);
}

using span0d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 0>,
                              std::experimental::layout_right>;
using span1d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 1>,
                              std::experimental::layout_right>;
using span2d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 2>,
                              std::experimental::layout_right>;
using span3d_t =
    std::experimental::mdspan<real_t, std::experimental::dextents<size_t, 3>,
                              std::experimental::layout_right>;

using local_acc = sycl::local_accessor<real_t, 3>;

using extents_t =
    std::experimental::extents<std::size_t, std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent,
                               std::experimental::dynamic_extent>;
