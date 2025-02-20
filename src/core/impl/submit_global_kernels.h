#pragma once

#include "IAdvectorX.h"
#include "advectors.h"

// ==========================================
// ==========================================
inline sycl::event
submit_local_kernel(sycl::queue &Q, double *fdist_dev, double *scratch,
                    const Solver &solver, const size_t b0_size,
                    const size_t b0_offset, const size_t b2_size,
                    const size_t b2_offset, const size_t orig_w0,
                    const size_t w1, const size_t orig_w2,
                    WorkGroupDispatch wg_dispatch, const size_t n0,
                    const size_t n1, const size_t n2) {

    const auto w0 = sycl::min(orig_w0, b0_size);
    const auto w2 = sycl::min(orig_w2, b2_size);

    wg_dispatch.set_num_work_groups(b0_size, b2_size, 1, 1, w0, w2);
    auto const seq_size0 = wg_dispatch.s0_;
    auto const seq_size2 = wg_dispatch.s2_;
    auto const g0 = wg_dispatch.g0_;
    auto const g2 = wg_dispatch.g2_;

    const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    const sycl::range<3> local_size(w0, w1, w2);

    return Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(scratch, w0, w2, n1);
                const auto i1 = itm.get_local_id(1);
                const auto local_i0 = itm.get_local_id(0);
                const auto local_i2 = itm.get_local_id(2);
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
                        sycl::group_barrier(itm.get_group());
                    }   // end for ii2
                }   // end for ii0
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // submit_local_kernels
