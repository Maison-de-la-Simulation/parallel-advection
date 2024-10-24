#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          double* fdist_dev,
                          const Solver &solver) {
    auto const n0 = solver.p.n0;
    auto const n1 = solver.p.n1;
    auto const n2 = solver.p.n2;

    const sycl::range global_size{n0, n1, n2};
    const sycl::range local_size{1, n1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                mdspan3d_t fdist(fdist_dev, n0, n1, n2);
                const int i1 = itm.get_local_id(1);
                const int i0 = itm.get_global_id(0);
                const int i2 = itm.get_global_id(2);

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                slice_ftmp[i1] = solver(i0, i1, i2, slice);

                sycl::group_barrier(itm.get_group());

                fdist(i0, i1, i2) = slice_ftmp[i1];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}
