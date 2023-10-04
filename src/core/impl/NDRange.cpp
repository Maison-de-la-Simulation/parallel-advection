#include "advectors.h"

sycl::event
AdvX::NDRange::operator()(sycl::queue &Q,
                          sycl::buffer<double, 2> &buff_fdistrib,
                          const ADVParams &params) noexcept {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const minRealVx = params.minRealVx;
    auto const dvx = params.dvx;
    auto const dt = params.dt;
    auto const realWidthX = params.realWidthX;

    const sycl::range<2> global_size{nvx, nx};
    const sycl::range<2> local_size{1, nx};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range{nx}, cgh);

        cgh.parallel_for(
            sycl::nd_range<2>{global_size, local_size},
            [=](sycl::nd_item<2> itm) {
                const int ix = itm.get_local_id(1);
                const int ivx = itm.get_global_id(0);

                // double const xFootCoord = displ(ix, ivx, params);
                double const xFootCoord = displ(
                    ix, ivx, minRealX, minRealVx, realWidthX, dx, dvx, dt);


                const int LeftDiscreteNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(LeftDiscreteNode,
                                                              minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                slice_ftmp[ix] = 0;   // initializing slice for each work item
                // double ftmp = 0.0;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1];
                }

                sycl::group_barrier(itm.get_group());
                fdist[ivx][ix] = slice_ftmp[ix];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}