#include "advectors.h"

sycl::event
AdvX::TwoDimWG::operator()(sycl::queue &Q,
                               sycl::buffer<double, 3> &buff_fdistrib,
                               const ADVParams &params) {

    auto const nx = params.nx;
    auto const nb = params.nb;
    auto const ns = params.ns;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_y = params.wg_size_y;
    auto const wg_size_x = params.wg_size_x;

    /* nb must be divisible by slice_size_dim_y */
    if(nb%wg_size_y != 0){
        throw std::invalid_argument("nb must be divisible by wg_size_y");
    }
    if(wg_size_y * nx > 6144){
        /* TODO: try with a unique allocation in shared memory and sequential iteration */
        throw std::invalid_argument("wg_size_y*nx must be < to 6144 (shared memory limit)");
    }

    const sycl::range nb_wg{nb/wg_size_y, 1, ns};
    const sycl::range wg_size{wg_size_y, params.wg_size_x, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* We use a 2D local accessor here */
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_y, nx), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{wg_size_y, nx, 1}, [&](sycl::h_item<3> it) {
                    const int ix = it.get_local_id(1);
                    const int iz = g.get_group_id(2);

                    const int local_nb = it.get_local_id(0);
                    const int ivx = wg_size_y * g.get_group_id(0) + local_nb;

                    double const xFootCoord = displ(ix, ivx, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[local_nb][ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[local_nb][ix] += coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.parallel_for_work_item(sycl::range{wg_size_y, nx, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int ix = it.get_local_id(1);
                                         const int iz = g.get_group_id(2);

                                         const int local_nb = it.get_local_id(0);
                                         const int ivx = wg_size_y * g.get_group_id(0) + local_nb;

                                         fdist[ivx][ix][iz] = slice_ftmp[local_nb][ix];
                                     });

            // g.async_work_group_copy(fdist.get_pointer()
            //                             + g.get_group_id(2)
            //                             + g.get_group_id(0) *ns*nx, /* dest */
            //                         slice_ftmp.get_pointer(), /* source */
            //                         nx*slice_size_dim_y, /* n elems */
            //                         ns  /* stride */
            // );
        });   // end parallel_for_work_group
    });       // end Q.submit
}
