#include "advectors.h"

sycl::event
AdvX::Exp5::operator()(sycl::queue &Q,
                          sycl::buffer<double, 3> &buff_fdistrib,
                          const ADVParams &params) {
    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    const sycl::range global_size{n0, 1         , wg_size_2_/n2};
    const sycl::range local_size {1 , wg_size_1_, wg_size_2_};

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);

        sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>{global_size, local_size},
            [=](auto itm) {
                const int i1 = itm.get_local_id(1);
                const int i0 = itm.get_global_id(0);
                const int i2 = itm.get_global_id(2);

                //for ii2 += wg_size_2_
                    //for ii1 += wg_size_1_


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
                    int idx_ipos1 = (n1 + ipos1 + k) % n1;

                    slice_ftmp[i1] += coef[k] * fdist[i0][idx_ipos1][i2];
                }

                sycl::group_barrier(itm.get_group());
                fdist[i0][i1][i2] = slice_ftmp[i1];
            }   // end lambda in parallel_for
        );      // end parallel_for nd_range
    });         // end Q.submit
}
