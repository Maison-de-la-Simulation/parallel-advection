#include "advectors.h"

sycl::event
AdvX::MemSafe::operator()(sycl::queue &Q,
                          sycl::buffer<double, 3> &buff_fdistrib,
                          const ADVParams &params) {
    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const nz = params.nz;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;
    
    if(nx > 6144){ //on A100 it breaks over 6144 double because of the size

    // if(nx*sizeof(double) > 49000){
        //TODO check MAX_SIZE du local_accessor
        // std::cerr << "Warning large size" << std::endl;
    }

    const sycl::range nb_wg{nvx, 1, nz};
    const sycl::range wg_size{1, params.wg_size, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);

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
                        inv_dx * (xFootCoord -
                                  coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[ix] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int idx_ipos1 = (nx + ipos1 + k) % nx;

                        slice_ftmp[ix] += coef[k] * fdist[ivx][idx_ipos1][iz];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            g.async_work_group_copy(fdist.get_pointer() + g.get_group_id(2) +
                                        g.get_group_id(0) * nz * nx, /* dest */
                                    slice_ftmp.get_pointer(), /* source */
                                    nx,                       /* n elems */
                                    nz                        /* stride */
            );
            // g.parallel_for_work_item(sycl::range{1, nx, 1},
            //                          [&](sycl::h_item<3> it) {
            //                              const int ix = it.get_local_id(0);
            //                              const int ivx = g.get_group_id(0);
            //                              const int iz = g.get_group_id(2);

            //                              fdist[ivx][ix][iz] = slice_ftmp[ix];
            //                          });
        });   // end parallel_for_work_group
    });       // end Q.submit
}
