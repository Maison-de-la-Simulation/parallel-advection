#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          sycl::buffer<double, 3> &buff_fdistrib,
                          const ADVParams &params) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range global_size{ny, nx, ny1};
    const sycl::range local_size{1, nx, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                const int ix = itm.get_local_id(1);
                const int iy = itm.get_global_id(0);
                const int iz = itm.get_global_id(2);

                double const xFootCoord = displ(ix, iy, params);

                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
                                                              minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;

                slice_ftmp[ix] = 0;   // initializing slice for each work item
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    slice_ftmp[ix] += coef[k] * fdist[iy][idx_ipos1][iz];
                }

                sycl::group_barrier(itm.get_group());
                fdist[iy][ix][iz] = slice_ftmp[ix];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}
