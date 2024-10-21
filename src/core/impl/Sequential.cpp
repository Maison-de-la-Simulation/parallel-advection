#include "advectors.h"
#include <cstddef>

sycl::event
AdvX::Sequential::operator()([[maybe_unused]] sycl::queue &Q,
                             sycl::buffer<double, 3> &buff_fdistrib,
                             const ADVParams &params) {
    auto const n1 = params.n1;
    auto const n0 = params.n0;
    auto const n2 = params.n2;
    auto const minRealX = params.minRealX;
    auto const dx = params.dx;
    auto const inv_dx = params.inv_dx;

    std::vector<double> slice_ftmp(n1);
    sycl::host_accessor fdist(buff_fdistrib, sycl::read_write);

    for (size_t i2 = 0; i2 < n2; ++i2) {
        for (size_t iv = 0; iv < n0; ++iv) {
            for (size_t iix = 0; iix < n1; ++iix) {
                // slice_x[iix] = fdist[iix][iv];
                slice_ftmp[iix] = 0;
            }

            // For each x with regards to current
            for (size_t i1 = 0; i1 < n1; ++i1) {

                double const xFootCoord = displ(i1, iv, params);

                const int leftNode =
                    sycl::floor((xFootCoord - minRealX) * inv_dx);

                const double d_prev1 =
                    LAG_OFFSET + inv_dx * (xFootCoord - coord(leftNode,
                                                              minRealX, dx));

                auto coef = lag_basis(d_prev1);

                const int ipos1 = leftNode - LAG_OFFSET;
                // double ftmp = 0.;
                for (auto k = 0; k <= LAG_ORDER; k++) {
                    int idx_ipos1 = (n1 + ipos1 + k) % n1;
                    // ftmp += coef[k] * slice_x[idx_ipos1];
                    slice_ftmp[i1] += coef[k] * fdist[idx_ipos1][iv][n2];
                }

                // fdist[i1][iv] = ftmp;
            }   // end for X

            for (size_t iix = 0; iix < n1; ++iix) {
                fdist[iix][iv][n2] = slice_ftmp[iix];
            }

        }   // end for Vx
    }       // end for z

    // returning empty submit to avoid warning and undefined behavior
    return Q.submit([&](sycl::handler &cgh) {
        // sycl::accessor fdist(buff_fdistrib, cgh, sycl::read_write);
        // sycl::local_accessor<double, 1> slice_ftmp(sycl::range<1>(n1), cgh);
        cgh.single_task([=]() {

        });   // end parallel_for 1 iter
    });       // end Q.submit
}
