#include "IAdvectorX.h"
#include "advectors.h"

// void
// print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
//     if (lvl == 0)
//         std::cout << "--------------------------------" << std::endl;
//     std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
//               << r.get(2) << "}" << std::endl;
// }

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::actual_advection(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver,
                                   const sycl::range<3> &global_size,
                                   const sycl::range<3> &local_size,
                                   const BlockingDispatch1D &block_n0,
                                   const BlockingDispatch1D &block_n2) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    auto const w0 = local_size.get(0);
    auto const w1 = local_size.get(1);
    auto const w2 = local_size.get(2);

    auto const n0_batch_size = block_n0.batch_size_;
    auto const n0_offset = block_n0.offset_;

    auto const n2_batch_size = block_n2.batch_size_;
    auto const n2_offset = block_n2.offset_;

    // const sycl::range global_size{n0_batch_size, w1, n2_batch_size};
    // const sycl::range local_size{w0, w1, w2};
    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 3> scratch(sycl::range(w0, w2, n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(scratch.get_pointer(), w0, w2, n1);

                const int local_i0 = itm.get_local_id(0);
                const int i0 = itm.get_global_id(0) + n0_offset;

                const int i1 = itm.get_local_id(1);

                const int local_i2 = itm.get_local_id(2);
                const int i2 = itm.get_global_id(2) + n2_offset;

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                    scr(local_i0, local_i2, ii1) = solver(slice, i0, ii1, i2);
                }

                sycl::group_barrier(itm.get_group());

                for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                    fdist(i0, ii1, i2) = scr(local_i0, local_i2, ii1);
                }
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}   // actual_advection

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::operator()(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver) {

    // Create copies of the blocking configurations
    BlockingDispatch1D blocks_d0 =
        bconf_d0_.last_dispatch_;   // Use last dispatch for d0 initially
    BlockingDispatch1D blocks_d2 =
        bconf_d2_.last_dispatch_;   // Use last dispatch for d2 initially

    blocks_d0.batch_size_ = max_batchs_x_;
    blocks_d2.batch_size_ = max_batchs_yz_;

    sycl::event last_event;

    for (size_t i0_batch = 0; i0_batch < bconf_d0_.n_batch_; ++i0_batch) {
        bool last_i0 = (i0_batch == bconf_d0_.n_batch_ - 1);
        blocks_d0.set_offset(
            i0_batch);   // Calculate the offset for the current batch

        for (size_t i2_batch = 0; i2_batch < bconf_d2_.n_batch_; ++i2_batch) {
            bool last_i2 = (i2_batch == bconf_d2_.n_batch_ - 1);
            blocks_d2.set_offset(
                i2_batch);   // Calculate the offset for the current batch

            // Select the correct batch dispatch based on whether we're in the
            // last batch of a dimension
            BlockingDispatch1D &dispatch_d0 =
                last_i0 ? bconf_d0_.last_dispatch_ : blocks_d0;
            BlockingDispatch1D &dispatch_d2 =
                last_i2 ? bconf_d2_.last_dispatch_ : blocks_d2;

            // Select the correct work-group dispatch
            WgDispatch wg = wg_dispatch_.normal_dispatch_;
            if (last_i0 && last_i2) {
                wg = wg_dispatch_.last_dispatch_d0_d2_;
            } else if (last_i0) {
                wg = wg_dispatch_.last_dispatch_d0_;
            } else if (last_i2) {
                wg = wg_dispatch_.last_dispatch_d2_;
            }

            // Compute global size based on batch sizes
            sycl::range<3> global_size(dispatch_d0.batch_size_, wg.w1_,
                                       dispatch_d2.batch_size_);
            sycl::range<3> local_size = wg.range();

            // If it's the last kernel submission, don't wait
            if (last_i0 && last_i2) {
                last_event =
                    actual_advection(Q, fdist_dev, solver, global_size,
                                     local_size, dispatch_d0, dispatch_d2);
            } else {
                actual_advection(Q, fdist_dev, solver, global_size, local_size,
                                 dispatch_d0, dispatch_d2)
                    .wait();
            }
        }
    }

    return last_event;
}
