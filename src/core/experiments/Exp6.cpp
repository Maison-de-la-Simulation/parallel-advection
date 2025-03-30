#include "advectors.h"

sycl::event
AdvX::Exp6::operator()(sycl::queue &Q, double *fdist_dev,
                       const Solver &solver) {
    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    const sycl::range global_size{n0, wg_size_1_, n2};
    const sycl::range local_size{1, wg_size_1_, wg_size_2_};

    auto const wg2 = wg_size_2_;
    auto const wg1 = wg_size_1_;

    auto scratch = scratch_;

    return Q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                mdspan3d_t scr(scratch, n0, n1, n2);

                const int i0 = itm.get_global_id(0);
                const int i1 = itm.get_local_id(1);
                const int i2 = itm.get_global_id(2);

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                // for(int ii2 = i2; ii2 < n2; ii2 += wg2){
                for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                    scr(i0, ii1, i2) = solver(slice, i0, ii1, i2);
                }
                // }

                sycl::group_barrier(itm.get_group());

                for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                    fdist(i0, ii1, i2) = scr(i0, ii1, i2);
                }
            }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}
