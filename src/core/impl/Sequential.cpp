#include "advectors.h"

sycl::event
AdvX::Sequential::operator()([[maybe_unused]] sycl::queue &Q,
                             sycl::buffer<double, 2> &buff_fdistrib,
                             const ADVParams &params) const {
    auto const nx = params.nx;
    size_t const nVx = params.nVx;
    auto const minRealx = params.minRealx;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    // returning to avoid warning and undefined behavior
    return Q.submit([&](sycl::handler &cgh) {
        sycl::accessor fdist(buff_fdistrib, cgh, sycl::read_write);

        cgh.single_task([=]() {

            double slice_x[nx];

            for (auto iv = 0; iv < nVx; ++iv) {

                // Memcopy slice x in contiguous memory
                for (int iix = 0; iix < nx; ++iix) {
                    slice_x[iix] = fdist[iix][iv];
                }

                // For each x with regards to current Vx
                for (int ix = 0; ix < nx; ++ix) {
                    double const xFootCoord = displ(ix, iv, params);

                    const int leftDiscreteCell =
                        sycl::floor((xFootCoord - minRealx) * inv_dx);

                    const double d_prev1 =
                        LAG_OFFSET +
                        inv_dx *
                            (xFootCoord - (minRealx + leftDiscreteCell * dx));

                    double coef[LAG_PTS];
                    lag_basis(d_prev1, coef);

                    const int ipos1 = leftDiscreteCell - LAG_OFFSET;
                    double ftmp = 0.;
                    for (int k = 0; k <= LAG_ORDER; k++) {
                        // int idx_ipos1 = (nx + ipos1 + k) % nx;
                        int idx_ipos1 = (nx + ipos1 + k) % nx;
                        ftmp += coef[k] * slice_x[idx_ipos1];
                    }

                    fdist[ix][iv] = ftmp;
                }   // end for X
            }       // end for Vx
        });         // end single_task
    });             // end Q.submit
}