#include "advectors.h"

sycl::event
AdvX::BasicRange::operator()(sycl::queue &Q,
                               sycl::buffer<double, 3> &buff_fdistrib,
                               const ADVParams &params) {
    auto const nx = params.nx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    Q.submit([&](sycl::handler &cgh) {
        auto fdist = buff_fdistrib.get_access<sycl::access::mode::read>(cgh);

        /* Using the preallocated global buffer */
        sycl::accessor ftmp(m_global_buff_ftmp, cgh, sycl::write_only,
                            sycl::no_init);

        cgh.parallel_for(buff_fdistrib.get_range(), [=](sycl::id<3> itm) {
            const int ix = itm[1];
            const int ivx = itm[0];
            const int iz = itm[2];

            double const xFootCoord = displ(ix, ivx, params);

            // Corresponds to the index of the cell to the left of footCoord
            const int leftNode =
                sycl::floor((xFootCoord - minRealX) * inv_dx);

            const double d_prev1 =
                LAG_OFFSET +
                inv_dx * (xFootCoord - coord(leftNode, minRealX, dx));

            auto coef = lag_basis(d_prev1);

            const int ipos1 = leftNode - LAG_OFFSET;

            ftmp[ivx][ix][iz] = 0;   // initializing slice for each work item
            for (int k = 0; k <= LAG_ORDER; k++) {
                int idx_ipos1 = (nx + ipos1 + k) % nx;

                ftmp[ivx][ix][iz] += coef[k] * fdist[ivx][idx_ipos1][iz];
            }

            // barrier
        });   // end parallel_for
    });       // end Q.submit

    return Q.submit([&](sycl::handler &cgh) {
        auto fdist = buff_fdistrib.get_access<sycl::access::mode::write>(cgh);
        auto ftmp =
            m_global_buff_ftmp.get_access<sycl::access::mode::read>(cgh);
        cgh.copy(ftmp, fdist);
    });   // end Q.submit
}
