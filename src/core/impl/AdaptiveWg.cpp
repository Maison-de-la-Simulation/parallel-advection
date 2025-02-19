#include "IAdvectorX.h"
#include "advectors.h"

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::submit_local_kernel(sycl::queue &Q, double *fdist_dev,
                                      const Solver &solver,
                                      const size_t &b0_size,
                                      const size_t b0_offset,
                                      const size_t &b2_size,
                                      const size_t b2_offset) {

    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    auto const w0 = local_size_.w0_;
    auto const w1 = local_size_.w1_;
    auto const w2 = local_size_.w2_;

    auto const seq_size0 = wg_dispatch_.s0_;
    auto const seq_size2 = wg_dispatch_.s2_;
    auto const g0 = wg_dispatch_.g0_;
    auto const g2 = wg_dispatch_.g2_;
    // wg_dispatch_.set_num_work_groups(n0, n2, dispatch_dim0_.n_batch_,
    //  dispatch_dim2_.n_batch_, w0, w2);
    const sycl::range<3> global_size(g0 * w0, w1, g2 * w2);
    const auto local_size = local_size_.range();

    std::cout << "in submit local kernel" << std::endl;
    print_range("global_size", global_size);
    print_range("local_size", local_size, 1);

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 3> scratch(sycl::range(w0, w2, n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(scratch.get_pointer(), w0, w2, n1);
                const auto i1 = itm.get_local_id(1);
                const auto local_i0 = itm.get_local_id(0);
                const auto local_i2 = itm.get_local_id(2);
                auto scratch_slice = std::experimental::submdspan(
                    scr, local_i0, local_i2, std::experimental::full_extent);

                if (itm.get_global_id(0) == 0 && itm.get_global_id(1) == 0 &&
                    itm.get_global_id(2) ==
                        0) {   // Print only once per workgroup
                    printf("seq_size0: %d, seq_size2: %d\n", seq_size0,
                           seq_size2);
                }

                for (size_t ii0 = 0; ii0 < seq_size0; ii0++) {
                    const auto global_i0 =
                        b0_offset + itm.get_global_id(0) + (ii0 * w0 * g0);
                    for (size_t ii2 = 0; ii2 < seq_size2; ii2++) {
                        const auto global_i2 =
                            b2_offset + itm.get_global_id(2) + (ii2 * w2 * g2);

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
                    sycl::group_barrier(itm.get_group());
                }   // end for ii0
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::operator()(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver) {

    sycl::event last_event;
    auto const &n_batch0 = dispatch_dim0_.n_batch_;
    auto const &n_batch2 = dispatch_dim2_.n_batch_;

    for (size_t i0_batch = 0; i0_batch < n_batch0; ++i0_batch) {
        bool last_i0 = (i0_batch == n_batch0 - 1);
        auto const offset_d0 = dispatch_dim0_.offset(i0_batch);

        for (size_t i2_batch = 0; i2_batch < n_batch2; ++i2_batch) {
            bool last_i2 = (i2_batch == n_batch2 - 1);
            auto const offset_d2 = dispatch_dim2_.offset(i2_batch);

            // Select the correct batch size
            size_t &batch_size_d0 = last_i0 ? dispatch_dim0_.last_batch_size_
                                            : dispatch_dim0_.batch_size_;
            size_t &batch_size_d2 = last_i2 ? dispatch_dim2_.last_batch_size_
                                            : dispatch_dim2_.batch_size_;

            if (last_i0 && last_i2) {
                last_event =
                    submit_local_kernel(Q, fdist_dev, solver, batch_size_d0,
                                        offset_d0, batch_size_d2, offset_d2);
            } else {
                submit_local_kernel(Q, fdist_dev, solver, batch_size_d0,
                                    offset_d0, batch_size_d2, offset_d2)
                    .wait();
            }
        }
    }

    return last_event;
}
