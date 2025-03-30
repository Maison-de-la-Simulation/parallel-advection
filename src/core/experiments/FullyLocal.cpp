#include "advectors.h"

sycl::event
AdvX::FullyLocal::operator()(sycl::queue &Q, double *fdist_dev,
                             const Solver &solver) {
    auto const n1 = solver.p.n1;
    auto const n0 = solver.p.n0;
    auto const n2 = solver.p.n2;

    // const sycl::range global_size{n0, 1         , 1};
    const sycl::range global_size{n0, wg_size_1_, n2};
    const sycl::range local_size{wg_size_0_, wg_size_1_, wg_size_2_};

    auto const wg2 = wg_size_2_;
    auto const wg1 = wg_size_1_;

    if (wg2 * n1 > MAX_LOCAL_ALLOC_) { //shouldn't trigger
        throw std::invalid_argument(
            "wg_size_0*n1 must be < to 6144 (local memory limit)");
    }

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 2> slice_ftmp(sycl::range<2>(wg2, n1),
                                                   cgh);

        cgh.parallel_for(sycl::nd_range<3>{global_size, local_size},
                         [=](auto itm) {
                             mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                             const int i0 = itm.get_global_id(0);
                             const int i1 = itm.get_local_id(1);
                             const int i2 = itm.get_global_id(2);

                             const int loc_i2 = itm.get_local_id(2);

                             auto slice = std::experimental::submdspan(
                                 fdist, i0, std::experimental::full_extent, i2);

                             for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                                 slice_ftmp[loc_i2][ii1] = solver(slice, i0, ii1, i2);
                             }

                             sycl::group_barrier(itm.get_group());

                             for (int ii1 = i1; ii1 < n1; ii1 += wg1) {
                                 fdist(i0, ii1, i2) = slice_ftmp[loc_i2][ii1];
                             }
                         }   // end lambda in parallel_for
        );   // end parallel_for nd_range
    });      // end Q.submit
}
