#include "x_advectors.h"

sycl::event
advector::x::BasicRange1D::operator()(sycl::queue &Q,
                                      sycl::buffer<double, 3> &buff_fdistrib,
                                      const ADVParams &params) noexcept {

    auto const nx = params.nx;
    auto const nvx = params.nvx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    // Q.submit([&](sycl::handler &cgh) {
    //     auto fdist = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);
    //     sycl::accessor<double, 3> ftmp(m_global_buff_ftmp, cgh,
    //                                    sycl::write_only, sycl::no_init);

    //     cgh.parallel_for(sycl::range<1>(nvx), [=](sycl::id<1> itm) {
    //         const int ivx = itm[0];

    //         for (int ix = 0; ix < nx; ++ix) {
    //             double const xFootCoord = displ(ix, ivx, params);

    //             // Corresponds to the index of the cell to the left of footCoord
    //             const int LeftDiscreteNode =
    //                 sycl::floor((xFootCoord - minRealx) * inv_dx);

    //             const double d_prev1 =
    //                 LAG_OFFSET + inv_dx * (xFootCoord - coord(LeftDiscreteNode,
    //                                                           minRealx, dx));

    //             auto coef = lag_basis(d_prev1);

    //             const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

    //             ftmp[0][ivx][ix] = 0;
    //             for (int k = 0; k <= LAG_ORDER; k++) {
    //                 int idx_ipos1 = (nx + ipos1 + k) % nx;

    //                 ftmp[0][ivx][ix] += coef[k] * fdist[0][ivx][idx_ipos1];

    //             }   // end for k
    //         }       // end for ix
    //         // barrier
    //     });   // end parallel_for
    // });       // end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
        auto ftmp =
            m_global_buff_ftmp.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(ftmp, fdist);
    });   // end Q.submit
}