#include "advectors.h"

sycl::event
AdvX::ReducedPrecision::operator()(sycl::queue &Q,
                                   sycl::buffer<double, 3> &buff_fdistrib,
                                   const ADVParams &params) {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const nz = params.nz;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{nvx, 1, nz};
    const sycl::range wg_size{1, params.wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* Float local accessor so we can put twice as much.
         We do trade numerical accuracy though */
        sycl::local_accessor<float, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int ivx = g.get_group_id(0);
                    const int iz = g.get_group_id(2);

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            /* Cannot use async_work_group_copy because float != double */
            g.parallel_for_work_item(sycl::range{1, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int ivx = g.get_group_id(0);
                                         const int iz = g.get_group_id(2);

                                         fdist[ivx][ix][iz] = slice_ftmp[ix];
                                     });
        });   // end parallel_for_work_group
    });       // end Q.submit
}
