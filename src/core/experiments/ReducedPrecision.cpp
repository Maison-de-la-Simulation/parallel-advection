#include "advectors.h"

sycl::event
AdvX::ReducedPrecision::operator()(sycl::queue &Q,
                                   double* fdist_dev,
                                   const ADVParams &params) {
    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range nb_wg{n0, 1, n2};
    const sycl::range wg_size{1, params.wg_size_1, 1};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        /* Float local accessor so we can put twice as much.
         We do trade numerical accuracy though */
        sycl::local_accessor<float, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for_work_group(nb_wg, wg_size, [=](sycl::group<3> g) {
            g.parallel_for_work_item(
                sycl::range{1, n1, 1}, [&](sycl::h_item<3> it) {
                    const int i1 = it.get_local_id(1);
                    const int i0 = g.get_group_id(0);
                    const int i2 = g.get_group_id(2);

                    double const xFootCoord = displ(i1, i0, params);

                    // index of the cell to the left of footCoord
                    const int leftNode =
                        sycl::floor((xFootCoord - minRealX) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

                    auto coef = lag_basis(d_prev1);

                    const int ipos1 = leftNode - LAG_OFFSET;

                    slice_ftmp[i1] = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        int id1_ipos = (n1 + ipos1 + k) % n1;

                        slice_ftmp[i1] += coef[k] * fdist[i0][id1_ipos][i2];
                    }
                });   // end parallel_for_work_item --> Implicit barrier

            /* Cannot use async_work_group_copy because float != double */
            g.parallel_for_work_item(sycl::range{1, n1, 1},
                                     [&](sycl::h_item<3> it) {
                                         const int i1 = it.get_local_id(1);
                                         const int i0 = g.get_group_id(0);
                                         const int i2 = g.get_group_id(2);

                                         fdist[i0][i1][i2] = slice_ftmp[i1];
                                     });
        });   // end parallel_for_work_group
    });       // end Q.submit
}
