#include "IAdvectorX.h"
#include "advectors.h"

// ==========================================
// ==========================================
sycl::event
AdvX::Nested::actual_advection(sycl::queue &Q, double *fdist_dev,
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

    auto const b0 = block_n0.batch_size_;
    auto const n0_offset = block_n0.offset_;

    auto const b2 = block_n2.batch_size_;
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

                const int i0 = itm.get_global_id(0);   // + n0_offset;
                const int i1 = itm.get_local_id(1);
                const int i2 = itm.get_global_id(2);   // + n2_offset;

                int local_i0 = itm.get_local_id(0);
                int local_i2 = itm.get_local_id(2);

                for (int ii0 = i0; ii0 < b0 + n0_offset; ii0 += w0) {

                    for (int ii2 = i2; ii2 < b2 + n2_offset; ii2 += w2) {

                        auto slice = std::experimental::submdspan(
                            fdist, ii0, std::experimental::full_extent, ii2);
                        for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                            scr(local_i0, local_i2, ii1) =
                                solver(slice, ii0, ii1, ii2);
                        }

                        sycl::group_barrier(itm.get_group());

                        for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                            fdist(ii0, ii1, ii2) = scr(local_i0, local_i2, ii1);
                        }
                        sycl::group_barrier(itm.get_group());
                    }
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


// class AdaptiveWg : public IAdvectorX {
//     using IAdvectorX::IAdvectorX;

//     AdaptiveWgDispatch wg_dispatch_;
//     BlockingConfig1D bconf_d0_;
//     BlockingConfig1D bconf_d2_;

//     const size_t max_batchs_x_ = 65536 - 1;
//     const size_t max_batchs_yz_ = 65536 - 1;

//     sycl::event actual_advection(sycl::queue &Q, double *fdist_dev,
//                                  const Solver &solver,
//                                  const sycl::range<3> &global_size,
//                                  const sycl::range<3> &local_size,
//                                  const BlockingDispatch1D &block_n0,
//                                  const BlockingDispatch1D &block_n2);

//   public:
//     sycl::event operator()(sycl::queue &Q, double *fdist_dev,
//                            const Solver &solver) override;

//     AdaptiveWg() = delete;

//     AdaptiveWg(const Solver &solver, sycl::queue q) {
//         const auto n0 = solver.params.n0;
//         const auto n1 = solver.params.n1;
//         const auto n2 = solver.params.n2;

//         bconf_d0_ = init_1d_blocking(n0, max_batchs_x_);
//         bconf_d2_ = init_1d_blocking(n2, max_batchs_yz_);

//         // SYCL query returns the size in bytes
//         auto max_elem_local_mem =
//             q.get_device().get_info<sycl::info::device::local_mem_size>() /
//             sizeof(double);

//         auto ideal_wg_dispatch =
//             compute_ideal_wg_size(solver.params.pref_wg_size, n0, n1, n2);

//         // Precompute adaptive work-group sizes
//         wg_dispatch_ = compute_adaptive_wg_dispatch(
//             ideal_wg_dispatch, bconf_d0_, bconf_d2_, n1, max_elem_local_mem);
//     }
// };


// using WgDispatch = sycl::range<3>;
// struct WgDispatch {
//     size_t w0_;
//     size_t w1_;
//     size_t w2_;

//     sycl::range<3> range() const { return sycl::range<3>{w0_, w1_, w2_}; }
//     inline size_t size() const {return w0_*w1_*w2_;}
// };

// struct AdaptiveWgDispatch {
//     WgDispatch normal_dispatch_;       // normal batches
//     WgDispatch last_dispatch_d0_;      // last batch in d0
//     WgDispatch last_dispatch_d2_;      // last batch in d2
//     WgDispatch last_dispatch_d0_d2_;   // last batch in
//                                        // both d0 and d2
// };

// struct BlockingDispatch1D {
//     size_t batch_size_;
//     size_t offset_;

//     void set_offset(size_t &i) { offset_ = batch_size_ * i; }
// };

// struct BlockingConfig1D {
//     size_t n_batch_;
//     size_t max_batch_size_;
//     BlockingDispatch1D last_dispatch_;
// };
