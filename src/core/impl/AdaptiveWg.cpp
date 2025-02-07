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
                                   const size_t &n0_batch_size,
                                   const size_t &n0_offset,
                                   const size_t &n2_batch_size,
                                   const size_t &n2_offset) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    auto const w0 = wg_dispatch_.w0_;
    auto const w1 = wg_dispatch_.w1_;
    auto const w2 = wg_dispatch_.w2_;

    const sycl::range global_size{n0_batch_size, w1, n2_batch_size};
    const sycl::range local_size{w0, w1, w2};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 3> scratch(sycl::range(w0, w2, n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(scratch.get_pointer(), w0, w2, n1);

                const int i0 = itm.get_global_id(0) + n0_offset;
                const int i1 = itm.get_local_id(1);
                const int i2 = itm.get_global_id(2) + n2_offset;

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                    scr(i0, ii1, i2) = solver(slice, i0, ii1, i2);
                }

                sycl::group_barrier(itm.get_group());

                for (int ii1 = i1; ii1 < n1; ii1 += w1) {
                    fdist(i0, ii1, i2) = scr(i0, ii1, i2);
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

    for (size_t i0_batch = 0; i0_batch < batchs_dispatch_d0_.n_batch_ - 1;
         ++i0_batch) {

        size_t n0_offset = (i0_batch * max_batchs_x_);

        for (size_t i2_batch = 2; i2_batch < batchs_dispatch_d2_.n_batch_ - 1;
            ++i2_batch) {

            size_t n2_offset = (i2_batch * max_batchs_yz_);

            actual_advection(Q, fdist_dev, solver, max_batchs_x_, n0_offset,
                             max_batchs_yz_, n2_offset)
                .wait();
            }
    }

    // return the last advection with the rest
    return actual_advection(
        Q, fdist_dev, solver, batchs_dispatch_d0_.last_batch_size_,
        batchs_dispatch_d0_.last_offset_, batchs_dispatch_d2_.last_batch_size_,
        batchs_dispatch_d2_.last_offset_);
}