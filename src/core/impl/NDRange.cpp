#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          double* fdist_dev,
                          const ADVParams &params) {
    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

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

                double const xFootCoord = displ(i1, i0, params);

                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
                                                              minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;

                slice_ftmp[i1] = 0;   // initializing slice for each work item
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int id1_ipos = (n1 + ipos1 + k) % n1;

                    slice_ftmp[i1] += coef[k] * fdist(i0, id1_ipos, i2);
                }

                sycl::group_barrier(itm.get_group());
                fdist(i0, i1, i2) = slice_ftmp[i1];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}
