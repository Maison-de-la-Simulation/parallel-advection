#include "advectors.h"

sycl::event
AdvX::Sequential::operator()([[maybe_unused]] sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params) {
    size_t const nx = params.nx;
    size_t const ny = params.ny;
    size_t const ny1 = params.ny1;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    std::vector<double> slice_ftmp(nx);
    sycl::host_accessor fdist(buff_fdistrib, sycl::read_write);

    for (size_t iy1 = 0; iy1 < ny1; ++iy1) {
        for (size_t iv = 0; iv < ny; ++iv) {
            for (size_t iix = 0; iix < nx; ++iix) {
                slice_ftmp[iix] = 0;
            }

            // For each x with regards to current
            for (size_t ix = 0; ix < nx; ++ix) {

                double const xFootCoord = displ(ix, iv, params);

                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
                                                              minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;
                for (auto k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (nx + ipos1 + k) % nx;
                    slice_ftmp[ix] += coef[k] * fdist[idx_ipos1][iv][ny1];
                }
            }   // end for X

            for (size_t iix = 0; iix < nx; ++iix) {
                fdist[iix][iv][ny1] = slice_ftmp[iix];
            }

        }   // end for Vx
    }       // end for z

    // returning empty submit to avoid warning and undefined behavior
    return Q.submit([&](sycl::handler &cgh) {
        cgh.single_task([=]() {});   // end parallel_for 1 iter
    });       // end Q.submit
}
