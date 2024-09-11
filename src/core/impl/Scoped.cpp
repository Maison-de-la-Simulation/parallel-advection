#include "advectors.h"

sycl::event
AdvX::Scoped::operator()(sycl::queue &Q,sycl::buffer<double, 3> &buff_fdistrib,
                         const ADVParams &params) {
    auto const nx = params.nx;
    auto const ny = params.ny;
    auto const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{ny, 1, ny1};
    const sycl::range wg_size{1, nx, 1};

    return Q.submit([&](sycl::handler &cgh) {
#ifdef SYCL_IMPLEMENTATION_ONEAPI   // for DPCPP
throw std::logic_error("Scoped kernel is not compatible with DPCPP");
#else   // for acpp
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

        cgh.parallel(nb_wg, wg_size, [=](auto g) {
                sycl::distribute_items_and_wait(g, [&](auto /*sycl::s_item<3>*/ it) {
                    const int ix = it.get_local_id(g, 1);
                    const int iy = g.get_group_id(0);
                    const int iz = g.get_group_id(2);

                    double const xFootCoord = displ(ix, iy, params);

                    // Corresponds to the index of the cell to
                    // the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[iy][idx_ipos1][iz];
                    }
                });   // end distribute items

                g.async_work_group_copy(
                    fdist.get_pointer() + g.get_group_id(2) +
                        g.get_group_id(0) * ny1 * nx,   // dest
                    slice_ftmp.get_pointer(),          // source
                    nx,                                /* n elems */
                    ny1                                 /* stride */
                );

                // sycl::distribute_items_and_wait(g, [&](auto it) {
                //     const int ix = it.get_local_id(g, 1);
                //     const int iy = g.get_group_id(0);
                //     const int iz = g.get_group_id(2);

                //     fdist[iy][ix][iz] = slice_ftmp[ix];
                // });
        }); // end parallel regions
#endif
    });// end Q.submit
}
