#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          real_t* fdist_dev,
                          const AdvectionSolver &solver) {
    auto const n0 = solver.params.n0;
    auto const n1 = solver.params.n1;
    auto const n2 = solver.params.n2;

    const sycl::range global_size{n0, n1, n2};
    const sycl::range local_size{1, n1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<real_t, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                span3d_t fdist(fdist_dev, n0, n1, n2);
                const int i1 = itm.get_local_id(1);
                const int i0 = itm.get_global_id(0);
                const int i2 = itm.get_global_id(2);

                auto slice = std::experimental::submdspan(
                    fdist, i0, std::experimental::full_extent, i2);

                slice_ftmp[i1] = solver(slice, i0, i1, i2);

                sycl::group_barrier(itm.get_group());

                slice(i1) = slice_ftmp[i1];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}
