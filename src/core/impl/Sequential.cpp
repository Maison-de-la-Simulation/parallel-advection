#include "advectors.h"

sycl::event
AdvX::Sequential::operator()([[maybe_unused]] sycl::queue &Q,
                             sycl::buffer<double, 2> &buff_fdistrib,
                             const ADVParams &params) {
    auto const nx = params.nx;
    size_t const nvx = params.nvx;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    std::vector<double> slice_ftmp(nx);
    sycl::host_accessor fdist(buff_fdistrib, sycl::read_write);

    for (auto iv = 0; iv < nvx; ++iv) {

        for (int iix = 0; iix < nx; ++iix) {
            // slice_x[iix] = fdist[iix][iv];
            slice_ftmp[iix] = 0;
        }

        // For each x with regards to current 
        for (auto ix = 0; ix < nx; ++ix) {

            double const xFootCoord = displ(ix, iv, params);

            const int LeftDiscreteNode =
                sycl::floor((xFootCoord - minRealX) * inv_dx);

            const double d_prev1 =
                LAG_OFFSET +
                inv_dx * (xFootCoord - coord(LeftDiscreteNode, minRealX, dx));

            auto coef = lag_basis(d_prev1);

            const int ipos1 = LeftDiscreteNode - LAG_OFFSET;
            // double ftmp = 0.;
            for (auto k = 0; k <= LAG_ORDER; k++) {
                int idx_ipos1 = (nx + ipos1 + k) % nx;
                // ftmp += coef[k] * slice_x[idx_ipos1];
                slice_ftmp[ix] += coef[k] * fdist[idx_ipos1][iv];
            }

            // fdist[ix][iv] = ftmp;
        }   // end for X

        for (int iix = 0; iix < nx; ++iix) {
            fdist[iix][iv] = slice_ftmp[iix];
        }

    }   // end for Vx

    // returning empty submit to avoid warning and undefined behavior
    return Q.submit([&](sycl::handler &cgh) {
        // sycl::accessor fdist(buff_fdistrib, cgh, sycl::read_write);
        // sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(nx), cgh);
        cgh.single_task([=]() {

        });   // end parallel_for 1 iter
    });       // end Q.submit
}
