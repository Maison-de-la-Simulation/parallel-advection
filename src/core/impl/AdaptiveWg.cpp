#include "IAdvectorX.h"
#include "advectors.h"

void
print_range(std::string_view name, sycl::range<3> r, bool lvl = 0) {
    if (lvl == 0)
        std::cout << "--------------------------------" << std::endl;
    std::cout << name << " : {" << r.get(0) << "," << r.get(1) << ","
              << r.get(2) << "}" << std::endl;
}

// ==========================================
// ==========================================
sycl::event
AdvX::AdaptiveWg::actual_advection(sycl::queue &Q, double *fdist_dev,
                                   const Solver &solver,
                                   const BlockingDispatch1D &block_n0,
                                   const BlockingDispatch1D &block_n2) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    auto const w0 = wg_dispatch_.w0_;
    auto const w1 = wg_dispatch_.w1_;
    auto const w2 = wg_dispatch_.w2_;

    auto const n0_batch_size = block_n0.batch_size_;
    auto const n0_offset = block_n0.offset_;

    auto const n2_batch_size = block_n2.batch_size_;
    auto const n2_offset = block_n2.offset_;

    const sycl::range global_size{n0_batch_size, w1, n2_batch_size};
    const sycl::range local_size{w0, w1, w2};

    // print_range("global_size", global_size);
    // print_range("local_size", local_size);
    // print_range("local_accessor_range", sycl::range(w0, w2, n1));

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



    auto const w0 = wg_dispatch_.w0_;
    auto const w1 = wg_dispatch_.w1_;
    auto const w2 = wg_dispatch_.w2_;
    // print_range("local_range", {w0, w1, w2});

    BlockingDispatch1D bdispatch_d0;
    BlockingDispatch1D bdispatch_d2;

    bdispatch_d0.batch_size_ = max_batchs_x_;
    bdispatch_d2.batch_size_ = max_batchs_yz_;

    for (size_t i0_batch = 0; i0_batch < bconf_d0_.n_batch_ - 1; ++i0_batch) {

        bdispatch_d0.set_offset(i0_batch);

        if (bconf_d2_.n_batch_ == 1) {
            actual_advection(Q, fdist_dev, solver, bdispatch_d0,
                             bconf_d2_.last_dispatch_)
                .wait();
        } else {
            for (size_t i2_batch = 0; i2_batch < bconf_d2_.n_batch_ - 1;
                 ++i2_batch) {
                bdispatch_d2.set_offset(i2_batch);
                actual_advection(Q, fdist_dev, solver, bdispatch_d0,
                                 bdispatch_d2)
                    .wait();
            }
        }

        // for (size_t i2_batch = 0; i2_batch < bconf_d2_.n_batch_ - 1;
        //      ++i2_batch) {

        //     bdispatch_d2.set_offset(i2_batch);
        //     // std::cout << "n2_offset" << n2_offset << std::endl;

        //     std::cout << "Batch i0_batch=" << i0_batch
        //               << " Offset=" << bdispatch_d0.offset_
        //               << " Size=" << bdispatch_d0.batch_size_ << std::endl;
        //     actual_advection(Q, fdist_dev, solver, bdispatch_d0, bdispatch_d2)
        //         .wait();
        // }

        //TODO add last batch_d0 dim!!!!
    }
    // return the last advection with the rest
    return actual_advection(Q, fdist_dev, solver, bconf_d0_.last_dispatch_,
                            bconf_d2_.last_dispatch_);
}