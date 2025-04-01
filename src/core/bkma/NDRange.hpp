#pragma once
#include <bkma_tools.hpp>

template <MemorySpace MemType, class MySolver, BkmaImpl Impl>
inline std::enable_if_t<Impl == BkmaImpl::NDRange, sycl::event>
submit_kernels(sycl::queue &Q, span3d_t data, const MySolver &solver,
               const size_t b0_size, const size_t b0_offset,
               const size_t b2_size, const size_t b2_offset,
               const size_t orig_w0, const size_t w1, const size_t orig_w2,
               WorkGroupDispatch wg_dispatch,
               span3d_t global_scratch = span3d_t{}) {

    const auto n0 = data.extent(0);
    const auto n1 = data.extent(1);
    const auto n2 = data.extent(2);

    const sycl::range global_size{n0, n1, n2};
    const sycl::range local_size{1, n1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<real_t, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for(sycl::nd_range<3>{global_size, local_size},
                         [=](auto itm) {
                             const int i1 = itm.get_local_id(1);
                             const int i0 = itm.get_global_id(0);
                             const int i2 = itm.get_global_id(2);

                             auto slice = std::experimental::submdspan(
                                 data, i0, std::experimental::full_extent, i2);

                             slice_ftmp[i1] = solver(slice, i0, i1, i2);

                             sycl::group_barrier(itm.get_group());

                             slice(i1) = slice_ftmp[i1];
                         }   // end lambda in parallel_for
        );                   // end parallel_for nd_range
    });                      // end Q.submit
}