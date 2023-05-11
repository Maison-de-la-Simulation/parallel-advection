#include "advectors.h"

sycl::event
AdvX::BasicRange1D::operator()(
    sycl::queue &Q, sycl::buffer<double, 2> &buff_fdistrib) const noexcept {

    auto const nx = m_params.nx;
    auto const nVx = m_params.nVx;
    auto const minRealx = m_params.minRealx;
    auto const dx = m_params.dx;
    auto const inv_dx = m_params.inv_dx;

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist =
            buff_fdistrib.get_access<sycl::access::mode::read_write>(cgh);
        sycl::accessor<double, 2> ftmp(m_global_buff_ftmp, cgh,
                                       sycl::read_write, sycl::no_init);

        cgh.parallel_for(sycl::range<1>(nVx), [=](sycl::id<1> itm) {
            const int ivx = itm[0];

            for (int ix = 0; ix < nx; ++ix) {
                double const xFootCoord = displ(ix, ivx);

                // Corresponds to the index of the cell to the left of footCoord
                const int LeftDiscreteNode =
                    sycl::floor((xFootCoord - minRealx) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(LeftDiscreteNode,
                                                              minRealx, dx));

                auto coef =  lag_basis(d_prev1);

                const int ipos1 = LeftDiscreteNode - LAG_OFFSET;

                ftmp[ivx][ix] = 0;
                for (int k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;

                    ftmp[ivx][ix] += coef[k] * fdist[ivx][idx_ipos1];

                }   // end for k
            }       // end for ix
            // barrier

            for (int i = 0; i < nx; ++i) {
                // fdist[i][ivx] = slice_ftmp[i];
                fdist[ivx][i] = ftmp[ivx][i];
            }
        });   // end parallel_for
    });       // end Q.submit
}