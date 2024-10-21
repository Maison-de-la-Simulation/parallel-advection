#include "advectors.h"

/* TODO: try with a unique allocation in local memory and sequential iteration */
sycl::event
AdvX::SeqTwoDimWG::operator()(sycl::queue &Q,
                              sycl::buffer<double, 3> &buff_fdistrib,
                              const ADVParams &params) {

    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    auto const wg_size_0 = params.wg_size_0;
    auto const wg_size_1 = params.wg_size_1;

    if(n0%wg_size_0 != 0){
        throw std::invalid_argument("n0 must be divisible by wg_size_0");
    }
    if(wg_size_0 * n1 > 6144){
        throw std::invalid_argument("wg_size_0*n1 must be < to 6144 (local memory limit)");
    }

    const sycl::range nb_wg{n0/wg_size_0, 1, n2};
    const sycl::range wg_size{wg_size_0, wg_size_1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* We use a 2D local accessor here */
        sycl::local_accessor<double, 2> slice_ftmp(
            sycl::range<2>(wg_size_0, n1), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {

            g.parallel_for_work_item(sycl::range{wg_size_0, n1, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int i1 = it.get_local_id(1);
                                         const int i2 = g.get_group_id(2);

                                         const int local_ny = it.get_local_id(0);
                                         const int i0 = wg_size_0 * g.get_group_id(0) + local_ny;

                                         slice_ftmp[local_ny][i1] = fdist[i0][i1][i2];
                                     });


            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    const int i1 = it.get_local_id(1);
                    const int i2 = g.get_group_id(2);

                    // int local_iy = it.get_global_id(0);
                    // get_local_id(0);
                    // const int i0 = wg_size_0 * g.get_group_id(0);

                    for (size_t iiy=0; iiy<wg_size_0; iiy++) {
                        const int i0 = wg_size_0 * g.get_group_id(0) + iiy;

                        double const xFootCoord = displ(i1, i0, params);

                        // index of the cell to the left of footCoord
                        const int leftNode =
                            sycl::floor((xFootCoord - minRealX) * inv_dx);

                        const double d_prev1 =
                            LAG_OFFSET +
                            inv_dx * (xFootCoord -
                                    coord(leftNode, minRealX, dx));

                        auto coef = lag_basis(d_prev1);

                        const int ipos1 = leftNode - LAG_OFFSET;

                        fdist[i0][i1][i2] = 0.;
                        for (int k = 0; k <= LAG_ORDER; k++) {
                            int id1_ipos = (n1 + ipos1 + k) % n1;

                            fdist[i0][i1][i2] += coef[k] * slice_ftmp[iiy][id1_ipos];
                        }
                    }

                });   // end parallel_for_work_item --> Implicit barrier


            // g.async_work_group_copy(fdist.get_pointer()
            //                             + g.get_group_id(2)
            //                             + g.get_group_id(0) *n2*n1, /* dest */
            //                         slice_ftmp.get_pointer(), /* source */
            //                         n1*slice_size_dim_y, /* n elems */
            //                         n2  /* stride */
            // );
        });   // end parallel_for_work_group
    });       // end Q.submit
}
